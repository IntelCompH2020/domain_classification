import logging
import pathlib
import sys
import os
from time import time

import pandas as pd
import numpy as np
import torch
from simpletransformers.classification import ClassificationModel
from sklearn import model_selection

# Mnemonics for values in column 'train_test'
TRAIN = 0
TEST = 1
UNUSED = -1


class CorpusClassifier(object):
    """
    A container of corpus classification methods
    """

    def __init__(self, df_dataset, path2transformers="."):
        """
        Initializes a preprocessor object

        Parameters
        ----------
        df_dataset : pandas.DataFrame
            Dataset with text and labels. It must contain at least two columns
            with names "text" and "labels", with the input and the target
            labels for classification.

        path2transformers : pathlib.Path or str, optional (default=".")
            Path to the folder that will store all files produced by the
            simpletransformers library.
            Default value is ".".

        Notes
        -----
        Be aware that the simpletransformers library produces several folders,
        with some large files. You might like to use a value of
        path2transformers other than '.'.
        """

        self.path2transformers = pathlib.Path(path2transformers)
        self.model = None
        self.df_dataset = df_dataset

        if 'labels' not in self.df_dataset:
            logging.error(f"-- -- Column 'labels' does not exist in the input "
                          "dataframe")
            sys.exit()

        return

    def train_test_split(self, max_imbalance=None, nmax=None, train_size=0.6,
                         random_state=None):
        """
        Split dataframe dataset into train an test datasets, undersampling
        the negative class

        Parameters
        ----------
        max_imbalance : int or None, optional (default=None)
            Maximum ratio negative vs positive samples. If the ratio in
            df_dataset is higher, the negative class is subsampled
            If None, the original proortions are preserved
        nmax : int or None (defautl=None)
            Maximum size of the whole (train+test) dataset
        train_size : float or int (default=0.6)
            Size of the training set.
            If float in [0.0, 1.0], proportion of the dataset to include in the
            train split.
            If int, absolute number of train samples.
        random_state : int or None (default=None)
            Controls the shuffling applied to the data before splitting.
            Pass an int for reproducible output across multiple function calls.

        Returns
        -------
        No variables are returned. The dataset dataframe in self.df_dataset is
        updated with a new columm 'train_test' taking values:
            0: if row is selected for training
            1: if row is selected for test
            -1: otherwise
        """

        l1 = sum(self.df_dataset['labels'])
        l0 = len(self.df_dataset) - l1

        # Selected dataset for training and testing. By default, it is equal
        # to the original dataset, but it might be reduced for balancing or
        # simplification purposes
        df_subset = self.df_dataset[['labels']]

        # Class balancing
        if (max_imbalance is not None and l0 > max_imbalance * l1):
            # Separate classes
            df0 = df_subset[df_subset['labels'] == 0]
            df1 = df_subset[df_subset['labels'] == 1]
            # Undersample majority class
            df0 = df0.sample(n=max_imbalance * l1)
            # Re-join dataframes
            df_subset = pd.concat((df0, df1))

        # Undersampling
        if nmax is not None and nmax < l0 + l1:
            df_subset = df_subset.sample(n=nmax)

        df_train, df_test = model_selection.train_test_split(
            df_subset, train_size=train_size, random_state=random_state,
            shuffle=True, stratify=None)

        # Marc train and test samples in the dataset.
        self.df_dataset['train_test'] = UNUSED
        self.df_dataset.loc[df_train.index, 'train_test'] = TRAIN
        self.df_dataset.loc[df_test.index, 'train_test'] = TEST

        return

    def load_model(self):
        """
        Loads an existing classification model

        Returns
        -------
        The loaded model is store in attribute self.model
        """

        # Create a TransformerModel
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logging.info(f"-- -- Cuda available: GPU will be used")
        else:
            logging.info(f"-- -- Cuda unavailable: no GPU will be used")

        # Expected location of the previously stored model.
        model_dir = self.path2transformers / "outputs" / "best_model"
        if not pathlib.Path.exists(model_dir):
            model_dir = self.path2transformers / "outputs"
        if not pathlib.Path.exists(model_dir):
            logging.error(f"-- No model available in {model_dir}")
            return

        # Load model
        self.model = ClassificationModel(
            'roberta', model_dir, use_cuda=cuda_available)

        return

    def train_model(self):
        """
        Train binary text classification model based on transformers

        Notes
        -----
        The use of simpletransformers follows the example code in
        https://towardsdatascience.com/simple-transformers-introducing-the-
        easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
        """

        # #################
        # Get training data
        if 'train_test' not in self.df_dataset:
            # Make partition if not available
            logging.warning(
                "-- -- Train test partition not available. A partition with "
                "default parameters will be generated")
            self.train_test_split()
        # Get training data (rows with value 1 in column 'train_test')
        # Note that we select the columns required for training only
        df_train = self.df_dataset[
            self.df_dataset.train_test == TRAIN][['text', 'labels']]

        # ##############
        # Classification

        # Create a TransformerModel
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logging.info(f"Cuda available: GPU will be used")
        else:
            logging.info(f"Cuda unavailable: training model without GPU")

        model_args = {
            'cache_dir': str(self.path2transformers / "cache_dir"),
            'output_dir': str(self.path2transformers / "outputs"),
            'best_model_dir': str(self.path2transformers / "outputs"
                                  / "best_model"),
            'tensorboard_dir': str(self.path2transformers / "runs"),
            'overwrite_output_dir': True}
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # FIXME: Base model 'roberta' should be configurable. Move it to
        #        the config file (parameters.default.yaml)
        self.model = ClassificationModel(
            'roberta', 'roberta-base', use_cuda=cuda_available,
            args=model_args)

        # Train the model
        logging.info(f"-- -- Training model with {len(df_train)} documents...")
        t0 = time()
        self.model.train_model(df_train)
        logging.info(f"-- -- Model trained in {time() - t0} seconds")

        return

    def eval_model(self, tag_score='score'):
        """
        Compute predictions of the classification model over the input dataset
        and compute performance metrics.

        Parameters
        ----------
        tag_score: str
            Prefix of the score names.
            The scores will be save in the columns of self.df_dataset
            containing these scores.

        Notes
        -----
        The use of simpletransformers follows the example code in
        https://towardsdatascience.com/simple-transformers-introducing-the-
        easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
        """

        # #############
        # Get test data

        # Check if a model has been trained
        if self.model is None:
            logging.error("-- -- A model must be trained before evalation")
            return

        # Get test data (rows with value 1 in column 'train_test')
        # Note that we select the columns required for training only
        df_test = self.df_dataset[
            self.df_dataset.train_test == TEST][['text', 'labels']]

        # #########################
        # Prediction and Evaluation

        # Evaluate the model
        logging.info(f"-- -- Testing model with {len(df_test)} documents...")
        t0 = time()
        result, model_outputs, wrong_predictions = self.model.eval_model(
            df_test)
        logging.info(f"-- -- Model tested in {time() - t0} seconds")

        # Add score columns if not available
        if f"{tag_score}_0" not in self.df_dataset:
            self.df_dataset[[f"{tag_score}_0", f"{tag_score}_1"]] = np.nan

        # Fill scores for the evaluated data
        self.df_dataset.loc[
            self.df_dataset['train_test'] == TEST,
            [f"{tag_score}_0", f"{tag_score}_1"]] = model_outputs

        # FIXME: Compute class predictions
        # FIXME: Add predictions to a new column in self.df_dataset.
        # FIXME: Map scores to probabilities.

        return result, wrong_predictions

    def sample(self, n_samples=5):
        """
        Returns a given number of samples for relevance feedback

        Parameters
        ----------
        n_samples : int, optional (default=5)
            Number of samples to return

        Returns
        -------
        df_out : pandas.dataFrame
            Selected samples
        """

        selected_docs = self.df_dataset.sample(n_samples)
        # FIXME: Try intelligent sample selection based on scores.

        return selected_docs

    def retrain_model(self):
        """
        Re-train the classifier model using annotations
        """

        # #################
        # Get training data

        # FIXME: Implement the data collection here:
        # He we should combine two colums from self.df_dataset
        #   - "labels", with the original labels used to train the first model
        #   - "annotations", with the new annotations
        # Notes:
        # Take into account that the annotation process could take place
        # iteratively, so this method could be called several times, each time
        # with some already used annotations and the new ones gathered from the
        # late human-annotation iteration. To help with this, you might use two
        # complementary columns from self.df_dataset
        #   - column 'date', with the annotation date
        #   - column 'used', marking with 1 those labels already used in
        #                    previous retrainings

        # ################
        # Model retraining
        # FIXME: Retrain the current model in self.model with the new labels

        # ################
        # Mark used labels
        if 'used' not in self.df_dataset:
            self.dataset[['used']] = 0
        # FIXME: Mark newly used labels with value 1 in column 'used'

        return
