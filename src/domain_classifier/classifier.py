import logging
import pathlib
import sys
from time import time
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import copy
from simpletransformers.classification import ClassificationModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from .custom_model import CustomModel
from sklearn import model_selection
from tqdm import tqdm

from transformers import logging as hf_logging

# Mnemonics for values in column 'train_test'
TRAIN = 0
TEST = 1
# Equivalent to NaN for the integer columns in self.df_dataset:
# (nan is not used because it converts the whole column to float)
UNUSED = -99

hf_logging.set_verbosity_error()


class CorpusClassifier(object):
    """
    A container of corpus classification methods
    """

    def __init__(self, df_dataset, path2transformers=".", use_cuda=True):
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

        use_cuda : boolean, optional (default=True)
            If true, GPU will be used, if available.

        Notes
        -----
        Be aware that the simpletransformers library produces several folders,
        with some large files. You might like to use a value of
        path2transformers other than '.'.
        """

        self.path2transformers = pathlib.Path(path2transformers)
        self.model = None
        self.df_dataset = df_dataset
        self.config = None

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("Cuda available: GPU will be used")
            else:
                logging.warning(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
                self.device = torch.device("cpu")
                logging.info(f"Cuda unavailable: training model without GPU")
        else:
            self.device = torch.device("cpu")
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
            # Recalculate number of class samples
            l1 = sum(df_subset["labels"])
            l0 = len(df_subset) - l1

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

    def load_model_config(self):
        """
        Load configuration for model.

        If there is no previous configuration, copy it from simpletransformers
        ClassificationModel and save it.
        """

        path2model_config = self.path2transformers / "config.json"

        # Save model config if not saved before
        if not path2model_config.exists():
            # Load TransformerModel
            logging.info("-- -- No available configuration. Loading "
                         "configuration from roberta model.")
            model = ClassificationModel(
                "roberta", "roberta-base", use_cuda=False)

            config = copy.deepcopy(model.config)

            # Save config
            config.to_json_file(path2model_config)
            logging.info("-- Model configuration saved")

            del model

        else:
            # Load config
            config = RobertaConfig.from_json_file(path2model_config)
            logging.info("-- Model configuration loaded from file")

        self.config = config

        return

    def load_model(self):
        """
        Loads an existing classification model

        Returns
        -------
        The loaded model is stored in attribute self.model
        """

        # Expected location of the previously stored model.
        model_dir = self.path2transformers / "best_model.pt"
        if not pathlib.Path.exists(model_dir):
            logging.error(f"-- No model available in {model_dir}")
            return

        # Load config
        self.load_model_config()

        # Load model
        self.model = CustomModel(self.config, self.path2transformers)
        self.model.load(model_dir)

        logging.info(f"-- Model loaded from {model_dir}")

        return

    def train_model(self, epochs=3, evaluate=True):
        """
        Train binary text classification model based on transformers

        Notes
        -----
        The use of simpletransformers follows the example code in
        https://towardsdatascience.com/simple-transformers-introducing-the-
        easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
        """

        # Create model directory
        self.path2transformers.mkdir(exist_ok=True)
        model_dir = self.path2transformers / "best_model.pt"

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
            self.df_dataset.train_test == TRAIN][['id', 'text', 'labels']]

        # TODO: set correct weight
        df_train["sample_weight"] = 1

        # ##############
        # Classification

        # Load config
        self.load_model_config()

        # Create model
        self.model = CustomModel(self.config, self.path2transformers)

        # # FIXME: Base model 'roberta' should be configurable. Move it to
        # #        the config file (parameters.default.yaml)

        # Best model selection
        best_epoch = 0
        best_result = 0
        best_predictions = None
        best_model = None

        # Train the model
        logging.info(f"-- -- Training model with {len(df_train)} documents...")
        t0 = time()
        for e in tqdm(range(epochs), desc="Train epoch"):

            # Train epoch
            epoch_loss, epoch_time = self.model.train_model(df_train)

            if evaluate:
                # #########################
                # Evaluation

                # Get test data (rows with value 1 in column 'train_test')
                # Note that we select the columns required for training only
                df_test = self.df_dataset[
                    self.df_dataset.train_test == TEST][
                        ['id', 'text', 'labels']]
                df_test["sample_weight"] = 1

                # Evaluate the model
                predictions, total_loss, result = self.model.eval_model(
                    df_test)

                if result["f1"] > best_result:
                    best_epoch = e
                    best_result = result["f1"]
                    best_predictions = predictions
                    best_model = copy.deepcopy(self.model)

        logging.info(f"-- -- Model trained in {time() - t0:.3f} seconds")

        if evaluate:
            self.model = best_model
            logging.info(f"-- Best model in epoch {best_epoch} with "
                         f"F1: {best_result:.3f}")

            # SCORES: Fill scores for the evaluated data
            self.df_dataset.loc[
                self.df_dataset['train_test'] == TEST,
                ["PUscore_0", "PUscore_1"]] = best_predictions

            # PREDICTIONS: Fill predictions for the evaluated data
            delta = predictions[:, 1] - predictions[:, 0]

            self.df_dataset["prediction"] = UNUSED
            self.df_dataset.loc[self.df_dataset['train_test'] == TEST,
                                "prediction"] = (delta > 0).astype(int)

            # Fill probabilistic predictions for the evaluated data
            # Scores are mapped to probabilities thoudh a logistic function.
            # FIXME: Check training loss in simpletransformers documentation or
            #        code, to see if logistic loss is appropriate here.
            self.df_dataset.loc[self.df_dataset['train_test'] == TEST,
                                f"prob_pred"] = 1 / (1 + np.exp(-delta))

        # Freeze middle layers
        self.model.freeze_encoder_layer()

        # Save model
        self.model.save(model_dir)
        logging.info(f"-- Model saved in {model_dir}")

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
            self.df_dataset.train_test == TEST][['id', 'text', 'labels']]

        if tag_score == "PUscore":
            df_test["sample_weight"] = 1
        else:
            df_test["sample_weight"] = 10

        # #########################
        # Prediction and Evaluation

        # Evaluate the model
        logging.info(f"-- -- Testing model with {len(df_test)} documents...")
        t0 = time()
        predictions, total_loss, result = self.model.eval_model(df_test)
        logging.info(f"-- -- Model tested in {time() - t0} seconds")

        # SCORES: Fill scores for the evaluated data
        self.df_dataset.loc[
            self.df_dataset['train_test'] == TEST,
            [f"{tag_score}_0", f"{tag_score}_1"]] = predictions

        # PREDICTIONS: Fill predictions for the evaluated data
        delta = predictions[:, 1] - predictions[:, 0]

        self.df_dataset["prediction"] = UNUSED
        self.df_dataset.loc[self.df_dataset['train_test'] == TEST,
                            "prediction"] = (delta > 0).astype(int)

        # Fill probabilistic predictions for the evaluated data
        # Scores are mapped to probabilities thoudh a logistic function.
        # FIXME: Check training loss in simpletransformers documentation or
        #        code, to see if logistic loss is appropriate here.
        self.df_dataset.loc[self.df_dataset['train_test'] == TEST,
                            f"prob_pred"] = 1 / (1 + np.exp(-delta))

        # TODO: redefine output of evaluation
        # result = {}
        wrong_predictions = []

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
        if not labels:
            logging.warning(f"-- Labels not confirmed")
            return
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
            & (self.df_dataset.learned == UNUSED)][['id', 'text', 'labels']]

        #  Get annotated samples already used for retraining
        df_clean_used = self.df_dataset[
            self.df_dataset.learned == 1][['id', 'text', 'labels']]

        #  Get new annotated samples, not used for retraining yet
        df_clean_new = self.df_dataset[
            self.df_dataset.learned == 0][['id', 'text', 'labels']]
        # ##################
        # Integrate datasets

        # FIXME: Change this by a more clever integration
        df_train = pd.concat([df_train_PU, df_clean_used, df_clean_new])

        # TODO: set correct weight
        df_train = df_clean_new
        df_train["sample_weight"] = 10

        # ##############
        # Classification

        # Train the model
        t0 = time()
        logging.info(f"-- -- Training model with {len(df_train)} documents...")
        self.model.train_model(df_train)
        logging.info(f"-- -- Model trained in {time() - t0:.3f} seconds")

        # ################
        # Update dataframe

        # Mark new annotations as used
        self.df_dataset.loc[self.df_dataset.learned == 0, 'learned'] = 1

        # Save model
        model_dir = self.path2transformers / "best_model.pt"
        self.model.save(model_dir)
        logging.info(f"-- Model saved in {model_dir}")

        return
