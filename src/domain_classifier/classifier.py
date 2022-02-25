import logging
import pathlib
import sys
import os
from time import time
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel
from sklearn import model_selection

# Mnemonics for values in column 'train_test'
TRAIN = 0
TEST = 1
# Equivalent to NaN for the integer columns in self.df_dataset:
# (nan is not used because it converts the whole column to float)
UNUSED = -99


class CustomDataset(Dataset):
    """
    Custom dataset to use with the custom model
    """
    def __init__(self, df):

        self.id = None
        self.text = None
        self.labels = None
        self.sample_weight = None

        for k, v in df.to_dict(orient="list").items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __getitem__(self, idx):
        item = {
            "id": self.id[idx],
            "text": self.text[idx],
            "sample_weight": self.sample_weight[idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.id)


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

        # Check if GPU is available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            logging.info(f"Cuda available: GPU will be used")
        else:
            logging.info(f"Cuda unavailable: training model without GPU")

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
        max_imbalance : int or float or None, optional (default=None)
            Maximum ratio negative vs positive samples. If the ratio in
            df_dataset is higher, the negative class is subsampled
            If None, the original proportions are preserved
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
            df0 = df0.sample(n=int(max_imbalance * l1))
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

    def create_data_loader(df, batch_size=8):
        """
        Creates a DataLoader from a DataFrame to train/eval model
        """

        df_set = CustomDataset(df)
        loader = DataLoader(
            dataset=df_set, batch_size=batch_size, shuffle=True, num_workers=0
        )

        return loader

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
            'roberta', 'roberta-base', use_cuda=self.cuda_available,
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

        # SCORES: Fill scores for the evaluated data
        self.df_dataset.loc[
            self.df_dataset['train_test'] == TEST,
            [f"{tag_score}_0", f"{tag_score}_1"]] = model_outputs

        # PREDICTIONS: Fill predictions for the evaluated data
        delta = model_outputs[:, 1] - model_outputs[:, 0]

        self.df_dataset["prediction"] = UNUSED
        self.df_dataset.loc[self.df_dataset['train_test'] == TEST,
                            "prediction"] = (delta > 0).astype(int)

        # Fill probabilistic predictions for the evaluated data
        # Scores are mapped to probabilities thoudh a logistic function.
        # FIXME: Check training loss in simpletransformers documentation or
        #        code, to see if logistic loss is appropriate here.
        self.df_dataset.loc[self.df_dataset['train_test'] == TEST,
                            f"prob_pred"] = 1 / (1 + np.exp(-delta))

        return result, wrong_predictions

    def AL_sample(self, n_samples=5):
        """
        Returns a given number of samples for active learning (AL)

        Parameters
        ----------
        n_samples : int, optional (default=5)
            Number of samples to return

        Returns
        -------
        df_out : pandas.dataFrame
            Selected samples
        """

        # Sample documents from the subset with predictions
        selected_docs = self.df_dataset.loc[
            self.df_dataset.prediction != UNUSED]

        if len(selected_docs) >= n_samples:
            selected_docs = selected_docs.sample(n_samples)
        else:
            logging.warning(
                "-- Not enough documents with predictions in the dataset")

        # FIXME: Try intelligent sample selection based on the scores or the
        #        probabilistic predictions in the self.df_dataset.

        return selected_docs

    def annotate(self, idx, labels, col='annotations'):
        """
        Annotate the given labels in the given positions

        Parameters
        ----------
        idx: list of int
            Rows to locate the labels.
        labels: list of int
            Labels to annotate
        col: str, optional (default = 'annotations')
            Column in the dataframe where the labels will be annotated. If it
            does not exist, it is created.
        """

        # Create annotation colum if it does not exist
        if col not in self.df_dataset:
            logging.info(
                f"-- -- Column {col} does not exist in dataframe. Added.")
            self.df_dataset[[col]] = UNUSED
        # Add labels to annotation columns
        self.df_dataset.loc[idx, col] = labels

        # Create colum of used labels if it does not exist
        if 'learned' not in self.df_dataset:
            self.df_dataset[['learned']] = UNUSED
        # Mark new labels as 'not learned' (i.e. not used by the learning
        # algorithm, yet.
        self.df_dataset.loc[idx, 'learned'] = 0

        # Add date to the dataframe
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y %H:%M:%S")
        self.df_dataset.loc[idx, 'date'] = date_str

        return

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
        #   - column 'learned', marking with 1 those labels already used in
        #                    previous retrainings

        # ####################
        # Get PU training data
        # Note that we select the columns required for training only
        # Note, also, that we exclude from the PU data the annotated labels
        df_train_PU = self.df_dataset[
            (self.df_dataset.train_test == TRAIN)
            & (self.df_dataset.learned == UNUSED)][['text', 'labels']]

        #  Get annotated samples already used for retraining
        df_clean_used = self.df_dataset[
            self.df_dataset.learned == 1][['text', 'labels']]

        #  Get new annotated samples, not used for retraining yet
        df_clean_new = self.df_dataset[
            self.df_dataset.learned == 0][['text', 'labels']]

        # ##################
        # Integrate datasets

        # FIXME: Change this by a more clever integration
        df_train = pd.concat([df_train_PU, df_clean_used, df_clean_new])

        # ##############
        # Classification

        # FIXME: replace by a more clever training
        # Create a TransformerModel
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
            'roberta', 'roberta-base', use_cuda=self.cuda_available,
            args=model_args)

        # Train the model
        t0 = time()
        logging.info(f"-- -- Training model with {len(df_train)} documents...")
        self.model.train_model(df_train)
        logging.info(f"-- -- Model trained in {time() - t0} seconds")

        # ################
        # Update dataframe

        # Mark new annotations as used
        self.df_dataset.loc[self.df_dataset.learned == 0, 'learned'] = 1

        return

