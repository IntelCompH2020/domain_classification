"""
Defines the main domain classification class

@author: J. Cid-Sueiro, J.A. Espinosa, A. Gallardo-Antolin
"""
import logging
import pathlib
from time import time
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import copy

from sklearn import model_selection
from tqdm import tqdm

from simpletransformers.classification import ClassificationModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.mpnet.configuration_mpnet import MPNetConfig

from transformers import logging as hf_logging

from .custom_model import CustomModel
# from .custom_model_roberta import RobertaCustomModel
# from .custom_model_mpnet import MPNetCustomModel
from . import metrics


# Mnemonics for values in column 'train_test'
TRAIN = 0
TEST = 1
# UNUSED: equivalent to NaN for the integer columns in self.df_dataset:
# (nan is not used because it converts the whole column to float)
UNUSED = -99

hf_logging.set_verbosity_error()


class CorpusClassifier(object):
    """
    A container of corpus classification methods
    """

    def __init__(self, df_dataset, model_type="mpnet",
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 path2transformers=".",
                 use_cuda=True):
        """
        Initializes a classifier object

        Parameters
        ----------
        df_dataset : pandas.DataFrame
            Dataset with text and labels. It must contain at least two columns
            with names "text" and "labels", with the input and the target
            labels for classification.

        model_type : str, optional (default="roberta")
            Type of transformer model.

        model_name : str, optional (default="roberta-base")
            Name of the simpletransformer model

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

        logging.info("-- Initializing classifier object")
        self.path2transformers = pathlib.Path(path2transformers)
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.df_dataset = df_dataset
        self.config = None

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("-- --Cuda available: GPU will be used")
            else:
                logging.warning(
                    "-- -- 'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
                self.device = torch.device("cpu")
                logging.info(
                    f"-- -- Cuda unavailable: model will be trained with CPU")
        else:
            self.device = torch.device("cpu")
            logging.info(f"-- -- Model will be trained with CPU")

        if 'labels' not in self.df_dataset:
            logging.error(" ")
            logging.error(f"-- -- Column 'labels' does not exist in the input "
                          "dataframe")
            logging.error(" ")

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

        logging.info("-- Selecting train and test ...")
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
        # The if model config file is available
        if not path2model_config.exists():
            logging.info("-- -- No available configuration. Loading "
                         f"configuration from {self.model_name} model.")

            # Load default config from the transformer model
            use_cuda = torch.cuda.is_available()
            model = ClassificationModel(
                self.model_type, self.model_name, use_cuda=use_cuda)
            self.config = copy.deepcopy(model.config)

            # Save config
            self.config.to_json_file(path2model_config)
            logging.info("-- -- Model configuration saved")

            del model

        else:
            # Load config
            if self.model_type == "roberta":
                self.config = RobertaConfig.from_json_file(path2model_config)
            elif self.model_type == "mpnet":
                self.config = MPNetConfig.from_json_file(path2model_config)
            else:
                logging.error("-- -- Config loading not available for "
                              + self.model_type)
                exit()

            logging.info("-- -- Model configuration loaded from file")

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
        self.model = CustomModel(self.config, self.path2transformers,
                                 self.model_type, self.model_name)
        # if self.model_type == 'roberta':
        #     self.model = RobertaCustomModel(
        #         self.config, self.path2transformers, self.model_name)
        # elif self.model_type == 'mpnet':
        #     self.model = MPNetCustomModel(
        #         self.config, self.path2transformers, self.model_name)

        self.model.load(model_dir)

        logging.info(f"-- Model loaded from {model_dir}")

        return

    def train_model(self, epochs=3, validate=True, freeze_encoder=True,
                    tag=""):
        """
        Train binary text classification model based on transformers

        Parameters
        ----------
        epochs : int, optional (default=3)
            Number of training epochs
        validate : bool, optional (default=True)
            If True, the model epoch is selected based on the F1 score computed
            over the test data. Otherwise, the model after the last epoch is
            taken
        freeze_encoder : bool, optional (default=True)
            If True, the embedding layer is frozen, so that only the
            classification layers is updated. This is useful to use
            precomputed embedings for large datasets.
        tag : str, optional (default="")
            A preffix that will be used for all result variables (scores and
            predictions) saved in the dataset dataframe

        Notes
        -----
        The use of simpletransformers follows the example code in
        https://towardsdatascience.com/simple-transformers-introducing-the-
        easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
        """

        logging.info("-- Training model...")

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
        self.model = CustomModel(self.config, self.path2transformers,
                                 self.model_type, self.model_name)

        # Freeze encoder layer if needed
        if freeze_encoder:
            self.model.freeze_encoder_layer()

        # Best model selection
        best_epoch = 0
        best_result = 0
        best_scores = None
        best_model = None

        # Train the model
        logging.info(f"-- Training model with {len(df_train)} documents...")
        t0 = time()
        for i, e in enumerate(tqdm(range(epochs), desc="Train epoch")):

            # Train epoch
            epoch_loss, epoch_time = self.model.train_model(
                df_train, self.device)

            logging.info(f"-- -- Epoch: {i} completed. loss: {epoch_loss:.3f}")

            if validate:
                # Get test data (rows with value 1 in column 'train_test')
                # Note that we select the columns required for training only
                df_test = self.df_dataset[
                    self.df_dataset.train_test == TEST][
                        ['id', 'text', 'labels']]
                df_test["sample_weight"] = 1

                # Evaluate the model
                # (Note that scores contain the (non-binary, non probabilistic)
                # classifier outputs).
                scores, total_loss, result = self.model.eval_model(
                    df_test, self.device)

                # Keep the best model
                if result["f1"] >= best_result:
                    best_epoch = e
                    best_result = result["f1"]
                    best_scores = scores
                    best_model = copy.deepcopy(self.model)

        logging.info(f"-- -- Model trained in {time() - t0:.3f} seconds")

        if validate:
            # Replace the last model by the best model
            self.model = best_model
            logging.info(f"-- Best model in epoch {best_epoch} with "
                         f"F1: {best_result:.3f}")

            # SCORES: Fill scores for the evaluated data
            test_rows = self.df_dataset['train_test'] == TEST
            self.df_dataset.loc[
                test_rows, [f"{tag}_score0", f"{tag}_score1"]] = best_scores

            # PREDICTIONS: Fill predictions for the evaluated data
            delta = scores[:, 1] - scores[:, 0]
            # Column "prediction" stores the binary class prediction.
            self.df_dataset[f"{tag}_prediction"] = UNUSED
            self.df_dataset.loc[test_rows, f"{tag}_prediction"] = (
                delta > 0).astype(int)

            # Fill probabilistic predictions for the evaluated data
            # Scores are mapped to probabilities through a logistic function.
            # FIXME: Check training loss in simpletransformers documentation or
            #        code, to see if logistic loss is appropriate here.
            prob_preds = 1 / (1 + np.exp(-delta))
            self.df_dataset.loc[test_rows, f"{tag}_prob_pred"] = prob_preds
            # A duplicate of the probablitistic predictions is stored in a
            # column without the tag, to be identified by the sampler of the
            # active learning algorithm as the last computed scores
            self.df_dataset.loc[test_rows, f"prob_pred"] = prob_preds

        # Freeze middle layers
        # self.model.freeze_encoder_layer()

        # Save model
        self.model.save(model_dir)
        logging.info(f"-- Model saved in {model_dir}")

        return

    def eval_model(self, samples="train_test", tag=""):
        """
        Compute predictions of the classification model over the input dataset
        and compute performance metrics.

        Parameters
        ----------
        samples : str, optional (default="train_test")
            Samples to evaluate. If "train_test" only training and test samples
            are evaluated. Otherwise, all samples in df_dataset attribute
            are evaluated
        tag : str
            Prefix of the score and prediction names.
            The scores will be saved in the columns of self.df_dataset
            containing these scores.
        """

        # #############
        # Get test data

        # Check if a model has been trained
        if self.model is None:
            logging.error("-- -- A model must be trained before evalation")
            return

        # Get test data (rows with value 1 in column 'train_test')
        # Note that we select the columns required for training only
        if samples == 'train_test':
            train_test = ((self.df_dataset.train_test == TRAIN)
                          | (self.df_dataset.train_test == TEST))
            df_test = (
                self.df_dataset[train_test][['id', 'text', 'labels']].copy())
        elif samples == 'all':
            df_test = self.df_dataset[['id', 'text', 'labels']].copy()
        else:
            msg = "-- -- Unknown value of parameter 'samples'"
            logging.error(msg)
            raise ValueError(msg)

        # FIXME: This condition should not be applied as a function of the tag,
        #        because the tag name can be arbitrarily modified. Maybe this
        #        weight should be passed as parameter.
        if tag == "PU":
            df_test["sample_weight"] = 1
        else:
            df_test["sample_weight"] = 10

        # #########################
        # Prediction and Evaluation

        # Evaluate the model
        logging.info(f"-- -- Testing model with {len(df_test)} documents...")
        t0 = time()
        scores, total_loss, result = self.model.eval_model(
            df_test, self.device)
        logging.info(f"-- -- Model tested in {time() - t0} seconds")

        # Compute probabilistic predicitons
        # Scores are mapped to probabilities through a logistic function.
        delta = scores[:, 1] - scores[:, 0]
        prob_preds = 1 / (1 + np.exp(-delta))

        # Fill dataset with the evaluation data:
        if samples == 'train_test':
            # Scores
            self.df_dataset.loc[
                train_test, [f"{tag}_score_0", f"{tag}_score_1"]] = scores
            # Class predictions
            self.df_dataset[f"{tag}_prediction"] = UNUSED
            self.df_dataset.loc[train_test, f"{tag}_prediction"] = (
                delta > 0).astype(int)
            # Probabilistic predictions
            self.df_dataset.loc[train_test, f"{tag}_prob_pred"] = prob_preds

        elif samples == 'all':
            # Scores
            self.df_dataset[[f"{tag}_score_0", f"{tag}_score_1"]] = scores
            # Class predictions
            self.df_dataset[f"{tag}_prediction"] = (delta > 0).astype(int)
            # Probabilistic predictions
            self.df_dataset[f"{tag}_prob_pred"] = prob_preds

        # TODO: redefine output of evaluation
        # result = {}
        wrong_predictions = []

        return result, wrong_predictions

    def performance_metrics(self, tag="", verbose=True):
        """
        Compute performance metrics

        Parameters
        ----------
        true_label_name : str
            Name of the column tu be used as a reference for evaluation
        verbose : boolean, optional (default=True)
            If True, the output metrics are printed
        """

        # ################
        # Training metrics

        df_train = self.df_dataset[self.df_dataset.train_test == TRAIN]
        # df_test = self.df_dataset[self.df_dataset.train_test == TEST]

        pscores = df_train[f"{tag}_prob_pred"].to_numpy()
        preds = df_train[f"{tag}_prediction"].to_numpy()
        labels = df_train[true_label_name].to_numpy()

        if set(labels).issubset({0, 1}) and set(preds).issubset({0, 1}):
            bmetrics_train = metrics.binary_metrics(preds, labels)
            metrics.print_binary_metrics(
                bmetrics_train, title="-- -- Metrics based on TRAIN data:")

            roc_train = metrics.score_based_metrics(pscores, labels)
            # metrics.plot_score_based_metrics(pscores, labels)

        else:
            logging.warning(
                "-- -- There are no predictions for the training samples")
            bmetrics_train = None
            roc_train = None

        # ############
        # Test metrics
        df_test = self.df_dataset[self.df_dataset.train_test == TEST]

        pscores = df_test[f"{tag}_prob_pred"].to_numpy()
        preds = df_test[f"{tag}_prediction"].to_numpy()
        labels = df_test[true_label_name].to_numpy()

        if set(labels).issubset({0, 1}) and set(preds).issubset({0, 1}):
            bmetrics_test = metrics.binary_metrics(preds, labels)
            metrics.print_binary_metrics(
                bmetrics_test, title="-- -- Metrics based on TEST data:")

            roc_test = metrics.score_based_metrics(pscores, labels)
            # metrics.plot_score_based_metrics(pscores, labels)

        else:
            logging.warning(
                "-- -- There are no predictions for the test samples")
            bmetrics_test = None
            roc_test = None

        return bmetrics_train, bmetrics_test, roc_train, roc_test

    def AL_sample(self, n_samples=5, sampler='extremes', p_ratio=0.8,
                  top_prob=0.1):
        """
        Returns a given number of samples for active learning (AL)

        Parameters
        ----------
        n_samples : int, optional (default=5)
            Number of samples to return
        sampler : str, optional (default="random")
            Sample selection algorithm.

            - If "random", samples are taken at random from all docs with
              predictions
            - If "extremes", samples are taken stochastically, but with
              documents with the highest or smallest probability scores are
              selected with higher probability.
            - If "full_rs", samples are taken at random from the whole dataset
              for testing purposes. Half samples are taken at random from the
              train-test split, while the rest is taken from the other
              documents

        p_ratio : float, optional (default=0.8)
            Ratio of high-score samples. The rest will be low-score samples.
            (Only for sampler='extremes')
        top_prob : float, optional (default=0.1)
            (Approximate) probability of selecting the doc with the highest
            score in a single sampling. This parameter is used to control the
            randomness of the stochastic sampling: if top_prob=1, the highest
            score samples are taken deterministically. top_prob=0 is equivalent
            to random sampling.

        Returns
        -------
        df_out : pandas.dataFrame
            Selected samples

        Notes
        -----
        Besides the output dataframe, this method updates columns 'sampler' and
        'sampling_prob' from self.dataframe.
            - 'sampler' stores the sampling method that selected the doc.
            - 'sampling_prob' is the probability with which each doc was
        selected. This is approximate, since the sampling of multiple documents
        is done without replacement, but it is reasonably accurate if the
        population sizes are large enough.
        """

        # Select documents with predictions only
        selected_docs = self.df_dataset.loc[
            self.df_dataset.prediction != UNUSED]

        if sampler == 'full_rs':
            unused_docs = self.df_dataset.loc[
                self.df_dataset.prediction == UNUSED]
        else:
            # Select documents without annotations only
            if 'annotations' in selected_docs.columns:
                selected_docs = selected_docs.loc[
                    selected_docs.annotations == UNUSED]

        if len(selected_docs) < n_samples:
            logging.warning(
                "-- Not enough documents with predictions in the dataset")
            return selected_docs

        if 'sampler' not in self.df_dataset:
            self.df_dataset[['sampler']] = "unsampled"
        if 'sampling_prob' not in self.df_dataset:
            self.df_dataset[['sampling_prob']] = UNUSED

        if sampler == 'random':

            if len(selected_docs) > n_samples:
                # ##############################
                # Compute sampling probabilities
                p = 1 / len(selected_docs)   # Population size

                # ###########
                # Sample docs
                selected_docs = selected_docs.sample(n_samples)
                selected_docs['sampling_prob'] = p

        elif sampler == 'extremes':

            if len(selected_docs) > n_samples:
                # ##############################
                # Compute sampling probabilities

                # Number of positive an negative samples to take
                n_pos = int(p_ratio * n_samples)
                n_neg = n_samples - n_pos

                # Generate exponentially decreasing selection probabilities
                n_doc = len(selected_docs)
                p = (1 - top_prob) ** np.array(range(n_doc))
                p = p / np.sum(p)

                # ###########
                # Sample docs

                # Sample documents with the highest scores
                # Sort selected docs by score
                selected_docs = selected_docs.sort_values(
                    'prob_pred', axis=0, ascending=False)
                selected_docs['sampling_prob'] = p
                selected_pos = selected_docs.sample(
                    n=n_pos, replace=False, weights='sampling_prob', axis=0)

                # Sample documents with the smallest scores
                selected_docs = selected_docs.sort_values(
                    'prob_pred', axis=0, ascending=True)
                selected_docs['sampling_prob'] = p
                selected_neg = selected_docs.sample(
                    n=n_neg, replace=False, weights='sampling_prob', axis=0)

                # Join samples
                selected_docs = pd.concat((selected_pos, selected_neg))

        elif sampler == 'full_rs':

            # ##############################
            # Compute sampling probabilities

            # ###########
            # Sample docs

            n_half = n_samples // 2
            if len(selected_docs) > n_half:
                p_selected = 1 / len(selected_docs)
                selected_docs = selected_docs.sample(n_half)
                selected_docs['sampling_prob'] = p_selected
            if len(unused_docs) > n_samples - n_half:
                p_unused = 1 / len(unused_docs)
                unused_docs = unused_docs.sample(n_samples - n_half)
                unused_docs['sampling_prob'] = p_unused

            selected_docs = pd.concat((selected_docs, unused_docs))

        else:
            logging.warning(f"-- Unknown sampling algorithm: {sampler}")
            return None

        idx = selected_docs.index
        # Register sampling mode
        selected_docs.loc[idx, 'sampler'] = sampler
        # Register sampling probabilities
        cols = ['sampler', 'sampling_prob']
        self.df_dataset.loc[idx, cols] = selected_docs[cols]

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

    def update_annotations(self, df_annotations, label_name):
        """
        Updates self.df_dataset with the annotation data and metadata in
        the input dataframe.

        Parameters
        ----------
        df_annotations : pandas.dataFrame
            A dataframe of annotations.
        label_name : Name of the column containing the class annotations
        """

        # Select from the input dataframe the columns related to annotations
        valid_cols = ['id', label_name, 'sampler', 'sampling_prob', 'date',
                      'train_test']
        cols = [c for c in valid_cols if c in self.df_dataset.columns]
        df_annotations = df_annotations[cols]

        # Remove rows that are not in the current dataset
        df_annotations = df_annotations.loc[
            df_annotations['id'].isin(self.df_dataset['id'])]

        # Create columns that did not exist
        new_cols = [c for c in df_annotations.cols if c not in self.df_dataset]
        for col in new_cols:
            self.df_dataset[col] = UNUSED

        # Merge into the current dataset
        self.df_dataset = self.df_dataset.set_index('id').update(
            df_annotations.set_index('id'), join='left', overwrite=True)

        return

    def retrain_model(self, freeze_encoder=True):
        """
        Re-train the classifier model using annotations
        """

        # #################
        # Get training data

        # FIXME: Implement the data collection here:
        # We should combine two colums from self.df_dataset
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

        if freeze_encoder:
            self.model.freeze_encoder_layer()

        # Train the model
        t0 = time()
        logging.info(f"-- -- Training model with {len(df_train)} documents...")
        self.model.train_model(df_train, self.device)
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
