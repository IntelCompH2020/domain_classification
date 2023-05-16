"""
Defines the main domain classification class

@author: J. Cid-Sueiro, J.A. Espinosa, A. Gallardo-Antolin, T.Ahlers
"""
import logging
import pathlib
from time import time
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.utils.data as data

import copy
from sklearn import model_selection
from tqdm import tqdm
import collections.abc

from simpletransformers.classification import ClassificationModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.mpnet.configuration_mpnet import MPNetConfig
from transformers import logging as hf_logging

# Local imports
from .custom_model import CustomModel
from .custom_model_mlp import MLP
from .custom_model_mlp import CustomDatasetMLP
from .inference import PandasModifier
from .inference import DataHandler
# from .custom_model_roberta import RobertaCustomModel
# from .custom_model_mpnet import MPNetCustomModel
from . import metrics

# Hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

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
        self.best_epoch = 15

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("-- -- Cuda available: GPU will be used")
            else:
                logging.warning(
                    "-- -- 'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False.")
                self.device = torch.device("cpu")
                logging.info(
                    f"-- -- Cuda unavailable: model will be trained with CPU")
        else:
            self.device = torch.device("cpu")
            logging.info(f"-- -- Model will be trained with CPU")

        if 'labels' not in self.df_dataset:
            logging.warning(" ")
            logging.warning(f"-- -- Column 'labels' does not exist in the "
                            "input dataframe")
            logging.warning(" ")

        return

    def _initialize_annotations(self, annot_name='annotations'):
        """
        Inititializs the annotation table before filling with annotations

        Parameters
        ----------
        annot_name : str, optional (default='annotations')
            Name of the column containing the class annotations
        """

        # Initialize columns
        if annot_name not in self.df_dataset:
            self.df_dataset[[annot_name]] = UNUSED
        if 'sampler' not in self.df_dataset:
            self.df_dataset[['sampler']] = "unsampled"
        if 'sampling_prob' not in self.df_dataset:
            self.df_dataset[['sampling_prob']] = UNUSED
        if 'date' not in self.df_dataset:
            self.df_dataset[['date']] = ""
        if 'train_test' not in self.df_dataset:
            self.df_dataset[['train_test']] = UNUSED
        if 'learned' not in self.df_dataset:
            self.df_dataset[['learned']] = UNUSED

        return

    def num_annotations(self):
        """
        Return the number of manual annotations available
        """

        df_annot = self.df_dataset[self.df_dataset.sampler != 'unsampled']
        n_labels = len(df_annot)
        n_train = sum(df_annot.train_test == TRAIN)
        n_test = sum(df_annot.train_test == TEST)
        n_unused = sum(df_annot.train_test == UNUSED)

        return n_labels, n_train, n_test, n_unused

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
            model = ClassificationModel(self.model_type, self.model_name,
                                        use_cuda=use_cuda)
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
        self.model.load(model_dir)

        logging.info(f"-- Model loaded from {model_dir}")

        return

    def _train_model(self, epochs=3, validate=True, freeze_encoder=True,
                     tag="", batch_size=8):
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
            classification layers are updated. This is useful to use
            precomputed embeddings for large datasets.
        tag : str, optional (default="")
            A preffix that will be used for all result variables (scores and
            predictions) saved in the dataset dataframe
        batch_size : int, optiona (default=8)
            Batch size
        """

        logging.info("-- Training model...")

        # Freeze encoder layer if needed
        if freeze_encoder:
            self.model.freeze_encoder_layer()

        # Get training data (rows with value 1 in column 'train_test')
        # Note that we select the columns required for training only
        df_train = self.df_dataset[
            self.df_dataset.train_test == TRAIN][[
                'id', 'text', 'labels', 'sample_weight']]

        # Best model selection
        if validate:
            # Get test data (rows with value 1 in column 'train_test')
            # Note that we select the columns required for training only
            df_test = self.df_dataset[
                self.df_dataset.train_test == TEST][[
                    'id', 'text', 'labels', 'sample_weight']]

        # Best model selection
        if validate:
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
                df_train, device=self.device, batch_size=batch_size)

            logging.info(f"-- -- Epoch: {i} completed. loss: {epoch_loss:.3f}")

            if validate:

                # Evaluate the model
                # (Note that scores contain the (non-binary, non probabilistic)
                # classifier outputs).
                scores, total_loss, result = self.model.eval_model(
                    df_test, device=self.device, batch_size=batch_size)

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
            self.best_epoch = best_epoch
            logging.info(f"-- Best model in epoch {best_epoch} with "
                         f"F1: {best_result:.3f}")

            # SCORES: Fill scores for the evaluated data
            test_rows = self.df_dataset['train_test'] == TEST
            self.df_dataset.loc[
                test_rows, [f"{tag}_score_0", f"{tag}_score_1"]] = best_scores

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
            # A duplicate of the predictions is stored in columns without the
            # tag, to be identified by the sampler of the active learning
            # algorithm as the last computed scores
            self.df_dataset[f"prediction"] = (
                self.df_dataset[f"{tag}_prediction"])
            self.df_dataset.loc[test_rows, f"prob_pred"] = prob_preds

        # Save model
        model_dir = self.path2transformers / "best_model.pt"
        self.model.save(model_dir)
        logging.info(f"-- Model saved in {model_dir}")

        return

    def train_model(self, epochs=3, validate=True, freeze_encoder=True,
                    tag="", batch_size=8):
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
        batch_size : int, optiona (default=8)
            Batch size
        """
        logging.info("-- Training model...")

        # ###############################
        # Initialize classification model

        # Create model directory
        self.path2transformers.mkdir(exist_ok=True)

        # Load config
        self.load_model_config()

        # Create model
        self.model = CustomModel(self.config, self.path2transformers,
                                 self.model_type, self.model_name)

        # #################
        # Get training data
        if 'train_test' not in self.df_dataset:
            # Make partition if not available
            logging.warning(
                "-- -- Train test partition not available. A partition with "
                "default parameters will be generated")
            self.train_test_split()
        self.df_dataset["sample_weight"] = 1

        # #####
        # Train

        self._train_model(
            epochs=epochs, validate=validate, freeze_encoder=freeze_encoder,
            tag=tag, batch_size=batch_size)

        return

    def eval_model(self, samples="train_test", tag="", batch_size=8):
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
        batch_size : int, optiona (default=8)
            Batch size
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

        # The following command is likely not needed, because it assigns the
        # default weight, 1, to all samples
        # The sample weights define the empirical risk, which is a weighted
        # sum. Since the weights used for training before and after annotation
        # are different, there are no "good" weights to use here.
        # We can used the last-used weights, but this information is not being
        # saved. It should be stored in self.df_dataset to be used here.
        df_test["sample_weight"] = 1

        # #########################
        # Prediction and Evaluation

        # Evaluate the model
        logging.info(f"-- -- Testing model with {len(df_test)} documents...")
        t0 = time()
        scores, total_loss, result = self.model.eval_model(
            df_test, device=self.device, batch_size=batch_size)
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

            # A duplicate of the predictions is stored in columns without the
            # tag, to be identified by the sampler of the active learning
            # algorithm as the last computed scores
            self.df_dataset.loc[train_test, "prediction"] = (
                delta > 0).astype(int)
            self.df_dataset.loc[train_test, "prob_pred"] = prob_preds

        elif samples == 'all':
            # Scores
            self.df_dataset[[f"{tag}_score_0", f"{tag}_score_1"]] = scores
            # Class predictions
            self.df_dataset[f"{tag}_prediction"] = (delta > 0).astype(int)
            # Probabilistic predictions
            self.df_dataset[f"{tag}_prob_pred"] = prob_preds

            # A duplicate of the predictions is stored in columns without the
            # tag, to be identified by the sampler of the active learning
            # algorithm as the last computed scores
            self.df_dataset["prediction"] = (
                self.df_dataset[f"{tag}_prediction"])
            self.df_dataset["prob_pred"] = prob_preds

        # TODO: redefine output of evaluation
        # result = {}
        wrong_predictions = []

        return result, wrong_predictions

    def performance_metrics(self, tag, true_label_name, subdataset,
                            pred_name=None, score_name=None, printout=True,
                            use_sampling_probs=True):
        """
        Compute performance metrics

        Parameters
        ----------
        tag : str in {'PU', 'PN'}
            Ettiquete of the model to be evaluated.
        true_label_name : str
            Name of the column tu be used as a reference for evaluation
        subdataset : str
            An indicator of the subdataset to be evaluated. It can take values
            'train', 'test', 'unused', 'notrain' (which uses train and test)
            and 'all' (which uses all data)
        printout : boolean, optional (default=True)
            If true, all metrics are printed (unless the roc values)
        use_sampling_probs: boolean, optional (default=True)
            If true, metrics are weighted by the (inverse) sampling
            probabilities, if available. If true, unweighted metrics are
            computed too, and saved in entry 'unweighted' of the output
            dictionaries, as complementary info.

        Returns
        -------
        bmetrics : dict
            A dictionary of binary metrics (i.e. metrics based on the binary
            labels and predictions only)
        roc : dict
            A dictionary of score-based metrics (i.e. metric based on the
            binary labels and predictions, and the scores (soft decistions)
            of the classifier)
        """

        # Make column names if not given
        if pred_name is None:
            pred_name = f"{tag}_prediction"
        if score_name is None:
            score_name = f"{tag}_prob_pred"

        # Select population
        if subdataset in {'train', 'test', 'unused'}:
            # Map subdataset to its integer code
            ttu = {'train': TRAIN, 'test': TEST, 'unused': UNUSED}[subdataset]
            df = self.df_dataset[self.df_dataset.train_test == ttu]
        elif subdataset == 'all':
            df = self.df_dataset.copy()
        elif subdataset == 'notrain':
            df = self.df_dataset[self.df_dataset.train_test != TRAIN]
        else:
            logging.error("Unknown subdataset")
            return None, None

        # Reduce dataset to samples with predictions and labels only
        l0 = len(df)
        df = df.loc[df[true_label_name].isin({0, 1})]
        df = df.loc[df[pred_name].isin({0, 1})]
        logging.info(f"-- -- Metrics based on {len(df)} out of {l0} samples")

        # Check if dataframe is not empty
        if len(df) == 0:
            logging.warning("-- -- No samples to evaluate")
            return None, None

        # Check if the selected labels and predictions contain labels in {0, 1}
        bmetrics, roc = None, None    # Default values

        # Extract predictions and labels
        pscores = df[score_name].to_numpy()
        preds = df[pred_name].to_numpy()
        labels = df[true_label_name].to_numpy()

        # Check if consistent sampling probabilities exist in df.
        if (use_sampling_probs
                and 'sampling_prob' in df
                and min(df['sampling_prob']) > 0
                and max(df['sampling_prob']) <= 1):

            # Compute weighted metrics
            p = df["sampling_prob"].to_numpy()
            bmetrics = metrics.binary_metrics(
                preds, labels, sampling_probs=p)
            roc = metrics.score_based_metrics(pscores, labels)

            # Compute and add unweighted metrics as a complementary entry
            # to the output dictionaries (this is just to facilitate the
            # comparison witht he weighted metrics)
            bmetrics['unweighted'] = metrics.binary_metrics(preds, labels)
            roc['unweighted'] = metrics.score_based_metrics(
                pscores, labels)

        else:
            # Compute unweighted metrics only
            bmetrics = metrics.binary_metrics(preds, labels)
            roc = metrics.score_based_metrics(pscores, labels)

        # Print
        if printout:
            title = f"{tag}_vs_{true_label_name}"
            metrics.print_metrics(bmetrics, roc, title=title, data=subdataset,
                                  print_unweighted=False)

        return bmetrics, roc

    def label2label_metrics(self, pred_name, true_label_name, subdataset,
                            printout=True, use_sampling_probs=True):
        """
        Compute binary performance metrics (i.e. metrics based on the binary
        labels and predictions only)

        Parameters
        ----------
        pred_name : str in {'PU', 'PN'}
            Ettiquete of the model to be evaluated.
        true_label_name : str
            Name of the column tu be used as a reference for evaluation
        subdataset : str
            An indicator of the subdataset to be evaluated. It can take values
            'train', 'test' or 'unused'
        printout : boolean, optional (default=True)
            If true, all metrics are printed (unless the roc values)
        use_sampling_probs: boolean, optional (default=True)
            If true, metrics are weighted by the (inverse) sampling
            probabilities, if available. If true, unweighted metrics are
            computed too, and saved in entry 'unweighted' of the output
            dictionary, as complementary info.

        Returns
        -------
        bmetrics : dict
            A dictionary of binary metrics
        """

        # Select population
        if subdataset in {'train', 'test', 'unused'}:
            # Map subdataset to its integer code
            ttu = {'train': TRAIN, 'test': TEST, 'unused': UNUSED}[subdataset]
            df = self.df_dataset[self.df_dataset.train_test == ttu]
        elif subdataset == 'all':
            df = self.df_dataset.copy()
        elif subdataset == 'notrain':
            df = self.df_dataset[self.df_dataset.train_test != TRAIN]
        else:
            logging.error("Unknown subdataset")
            return None

        # Reduce dataset to samples with predictions and labels only
        l0 = len(df)
        df = df.loc[df[true_label_name].isin({0, 1})]
        df = df.loc[df[pred_name].isin({0, 1})]
        logging.info(f"-- -- Metrics based on {len(df)} out of {l0} samples")

        # Check if dataframe is not empty
        if len(df) == 0:
            logging.warning("-- -- No samples to evaluate")
            return None

        bmetrics = None    # Default
        # Extract predictions and labels
        preds = df[pred_name].to_numpy()
        labels = df[true_label_name].to_numpy()

        if (use_sampling_probs
                and 'sampling_prob' in df
                and min(df['sampling_prob']) > 0
                and max(df['sampling_prob']) <= 1):

            # Compute weighted metrics
            p = df["sampling_prob"].to_numpy()
            bmetrics = metrics.binary_metrics(
                preds, labels, sampling_probs=p)

            # Compute and add unweighted metrics as a complementary entry
            # to the output dictionaries (this is just to facilitate the
            # comparison witht he weighted metrics)
            bmetrics['unweighted'] = metrics.binary_metrics(preds, labels)

        else:
            # Compute unweighted metrics only
            bmetrics = metrics.binary_metrics(preds, labels)

        # Print
        if printout:
            title = f"{pred_name}_vs_{true_label_name}"
            metrics.print_metrics(bmetrics, title=title, data=subdataset,
                                  print_unweighted=False)

        return bmetrics

    def print_binary_metrics(self, bmetrics, tag=""):
        """
        Pretty-prints the given metrics

        Parameters
        ----------
        bmetrics : dict
            Dictionary of metrics (produced by the binary_metrics() method)
        title : str, optional (default="")
            Title to print as a header
        """

        metrics.print_binary_metrics(bmetrics, tag)

        return

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

        # Check sampler
        valid_samplers = ['random', 'extremes', 'full_rs', 'least_confidence']
        if sampler not in valid_samplers:
            logging.error(
                f'-- -- Sampler unknown: available options are '
                f'{", ".join(valid_samplers)}')

        # This is for backward compatibility
        if 'prediction' not in self.df_dataset:
            self.df_dataset['prediction'] = self.df_dataset['PU_prediction']

        # Documents used for train or test (validation)
        train_test = ((self.df_dataset.train_test == TRAIN)
                      | (self.df_dataset.train_test == TEST))
        selected_docs = self.df_dataset.loc[train_test]

        if sampler == 'full_rs':
            unused_docs = self.df_dataset.loc[~train_test]
        else:
            # Select documents without annotations only
            if 'annotations' in selected_docs.columns:
                selected_docs = selected_docs.loc[
                    selected_docs.annotations == UNUSED]

        if len(selected_docs) < n_samples:
            logging.warning(
                "-- Not enough documents with predictions in the dataset")
            return selected_docs

        # Initialize columns 'sampler' or 'sampling_prob' if they do not exist
        if 'sampler' not in self.df_dataset:
            self.df_dataset[['sampler']] = "unsampled"
        if 'sampling_prob' not in self.df_dataset:
            self.df_dataset[['sampling_prob']] = UNUSED

        if sampler == 'random':

            # Compute sampling probabilities
            p = 1 / len(selected_docs)   # Population size

            # Sample docs
            if len(selected_docs) > n_samples:
                selected_docs = selected_docs.sample(n_samples)

            # Assign sampling probabilities
            selected_docs['sampling_prob'] = p

        elif sampler == 'least_confidence':

            self.eval_model()

            df_train = self.df_dataset[self.df_dataset['train_test'] == 0]
            df_test = self.df_dataset[self.df_dataset['train_test'] == 1]
            train_ratio = len(df_train) / (len(df_train) + len(df_test))
            sample_size_train = int(np.round(n_samples * train_ratio))
            sample_size_test = n_samples - sample_size_train

            sort_idx_train = np.argsort(
                np.abs(df_train['prob_pred'].to_numpy() - 0.5))
            sort_idx_test = np.argsort(
                np.abs(df_test['prob_pred'].to_numpy() - 0.5))

            selected_docs = pd.concat(
                [df_train.iloc[sort_idx_train][:sample_size_train],
                 df_test.iloc[sort_idx_test][:sample_size_test]])

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

            else:
                # Compute sampling probabilities
                p = 1 / len(selected_docs)   # Population size
                # Assign sampling probabilities
                selected_docs['sampling_prob'] = p

        elif sampler == 'full_rs':

            # Compute sampling probabilities
            n_half = n_samples // 2
            p_selected = 1 / len(selected_docs)
            p_unused = 1 / len(unused_docs)

            # Sample docs
            if len(selected_docs) > n_half:
                selected_docs = selected_docs.sample(n_half)
            if len(unused_docs) > n_samples - n_half:
                unused_docs = unused_docs.sample(n_samples - n_half)

            # Assign sampling probabilities
            selected_docs['sampling_prob'] = p_selected
            unused_docs['sampling_prob'] = p_unused

            # Join samples from the train-test and the unused subdatasets
            selected_docs = pd.concat((selected_docs, unused_docs))

        else:
            logging.warning(f"-- Unknown sampling algorithm: {sampler}")
            return None

        # Register sampler and sampling probabilities into the dataset
        idx = selected_docs.index
        selected_docs.loc[idx, 'sampler'] = sampler
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

    def get_annotations(self, annot_name='annotations', include_text=True):
        """
        Returns the portion of self.dataset that contains annotated data

        Parameters
        ----------
        annot_name : str, optional (default='annotations')
            Name of the column in the pandas dataframe containing the
            annotations
        include_text : bool, optional (default=True)
            If true, the text of the annotated document is included in the
            output dataframe. This is usefull for a manual inspection of
            the annotations.

        Returns
        -------
        df_annotation : pandas.dataFrame
            The dataframe containing the annotations. All columns related
            to the annotation are returned.
        """

        # Extract label dataframe from the dataset.
        if include_text:
            cols = ['id', 'text', annot_name, 'sampler', 'sampling_prob',
                    'date', 'train_test']
        else:
            cols = ['id', annot_name, 'sampler', 'sampling_prob', 'date',
                    'train_test']
        cols = [c for c in cols if c in self.df_dataset.columns]

        # Identify annotated docs:
        annotated_docs = self.df_dataset['sampler'] != 'unsampled'
        df_annotations = self.df_dataset.loc[annotated_docs, cols]
        df_annotations.reset_index(drop=True, inplace=True)

        return df_annotations

    def update_annotations(self, df_annotations, annot_name='annotations'):
        """
        Updates self.df_dataset with the annotation data and metadata in
        the input dataframe.

        Parameters
        ----------
        df_annotations : pandas.dataFrame
            A dataframe of annotations.
        annot_name : str, optional (default='annotations')
            Name of the column containing the class annotations
        """

        self._initialize_annotations(annot_name=annot_name)

        # Select from the input dataframe the columns related to annotations
        valid_cols = ['id', annot_name, 'sampler', 'sampling_prob', 'date',
                      'train_test']

        cols = [c for c in valid_cols if c in self.df_dataset.columns]
        df_annotations = df_annotations[cols]

        # Remove rows that are not in the current dataset
        df_annotations = df_annotations.loc[
            df_annotations['id'].isin(self.df_dataset['id'])]

        # Merge into the current dataset
        self.df_dataset.set_index('id', inplace=True)
        self.df_dataset.update(
            df_annotations.set_index('id'), join='left', overwrite=True)

        # This is to go back to the original indices. This might be not
        # required. Likely, column 'id' could be used as index in the whole
        # class, but it would require some recoding. To be done.
        self.df_dataset.reset_index(inplace=True)

        return

    def retrain_model(self, freeze_encoder=True, batch_size=8, epochs=3,
                      annotation_gain=10):
        """
        Re-train the classifier model using annotations

        Parameters
        ----------
        epochs : int, optional (default=3)
            Number of training epochs
        freeze_encoder : bool, optional (default=True)
            If True, the embedding layer is frozen, so that only the
            classification layers is updated. This is useful to use
            precomputed embedings for large datasets.
        batch_size : int, optional (default=8)
            Batch size
        annotation_gain : int or float, optional (default=10)
            Relative value of an annotated sample with respect to a non-
            annotated one.
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
        #     previous retrainings

        # ####################
        # Get PU training data
        # Note that we select the columns required for training only

        # Training data not annotated (i.e., UNUSED in column 'learned')
        is_train = (self.df_dataset.train_test == TRAIN)

        if 'learned' in self.df_dataset:
            is_tr_unused = is_train & (self.df_dataset.learned == UNUSED)
            is_tr_used = is_train & (self.df_dataset.learned == 1)
            is_tr_new = is_train & (self.df_dataset.learned == 0)
        else:
            is_tr_unused = is_train
            is_tr_used = []
            is_tr_new = []

        self.df_dataset.loc[is_tr_unused, "sample_weight"] = 1
        self.df_dataset.loc[is_tr_used, "sample_weight"] = annotation_gain
        self.df_dataset.loc[is_tr_new, "sample_weight"] = annotation_gain

        self.df_dataset.loc[is_train, "labels"] = self.df_dataset.loc[
            is_train, "PUlabels"]
        if 'annotations' in self.df_dataset:
            self.df_dataset.loc[is_tr_used, "labels"] = self.df_dataset.loc[
                is_tr_used, "annotations"]
            self.df_dataset.loc[is_tr_new, "labels"] = self.df_dataset.loc[
                is_tr_new, "annotations"]

        # #######
        # Retrain

        self._train_model(
            epochs=3, validate=True, freeze_encoder=freeze_encoder, tag='PN',
            batch_size=batch_size)

        # ################
        # Update dataframe

        # Mark new annotations as used
        if 'learned' in self.df_dataset:
            self.df_dataset.loc[self.df_dataset.learned == 0, 'learned'] = 1

        return


class CorpusClassifierMLP(CorpusClassifier):
    """
    Defines the main domain classification class using embeddings

    @author: J. Cid-Sueiro, J.A. Espinosa, A. Gallardo-Antolin, T.Ahlers
    """

    def __init__(self, df_dataset, model_type="mpnet",
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 path2transformers=".", use_cuda=True):
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

        """

        # ['id', 'text', 'base_scores', 'PUlabels', 'labels', 'embeddings'],
        super().__init__(df_dataset, model_type=model_type,
                         model_name=model_name,
                         path2transformers=path2transformers,
                         use_cuda=use_cuda)

    def __sample_train_data(self, retrain=False):

        if not retrain:
            df_potential_train = (
                self.df_dataset[self.df_dataset['train_test'] == 0].copy())
        else:
            df_potential_train = self.df_dataset[
                (self.df_dataset['train_test'] == 0)
                & (self.df_dataset['annotations'] != UNUSED)].copy()
        df_positive = df_potential_train[df_potential_train['labels'] == 1]
        df_negative = df_potential_train[df_potential_train['labels'] == 0]
        df_majority = (df_positive if len(df_positive) >= len(df_negative)
                       else df_negative)
        df_minority = (df_positive if len(df_positive) < len(df_negative)
                       else df_negative)

        if len(df_minority) == 0:
            return[]

        return pd.concat([df_majority, df_minority.sample(len(df_majority),
                                                          replace=True)])

    def __sample_validation_data(self, retrain=False):
        """
        samples the validation data so that the ratio between
        positive and negative in the annoteated data is the same
        like the positie ration in the whole dataset applying the
        weak label threshold on the whole dataset

        Parameters
        ----------
        retrain :
            For training it takes just the data which is flagged as
            validation

        Returns
        -------
        dataframe with validation data

        """

        if not retrain:
            return self.df_dataset[self.df_dataset['train_test'] == 1].copy()
        df_potential_validation = self.df_dataset[
            (self.df_dataset['train_test'] == 1)
            & (self.df_dataset['annotations'] != UNUSED)].copy()

        if len(df_potential_validation['labels'].unique()) < 2:
            return df_potential_validation

        weak_label_threshold = df_potential_validation.groupby(
            ['labels']).mean()['base_scores'].mean()
        positive_ratio = (
            len(self.df_dataset[self.df_dataset['base_scores']
                                > weak_label_threshold])
            / len(self.df_dataset))

        df_positive = df_potential_validation[
            df_potential_validation['labels'] == 1]
        df_negative = df_potential_validation[
            df_potential_validation['labels'] == 0]

        n_pos = len(df_positive)
        n_neg = len(df_negative)
        bKeepPositive = (n_pos / len(df_potential_validation)) > positive_ratio
        df_keep = df_positive if bKeepPositive else df_negative
        df_sample = df_negative if bKeepPositive else df_positive

        if bKeepPositive:
            sample_count = int(np.round((n_pos / n_neg) / positive_ratio))
        else:
            sample_count = int(np.round((n_neg / n_pos) * positive_ratio))

        return pd.concat(
            [df_keep, df_sample.sample(sample_count, replace=True)])

    def train_model(self, epochs=3, validate=True, freeze_encoder=True,
                    tag="", batch_size=8):
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
        batch_size : int, optiona (default=8)
            Batch size
        """


        logging.info("-- Training model...")

        self._train_model()

        #self.load_model()
        #
        # df_train = self.__sample_train_data()
        # if len(df_train) == 0:
        #     logging.info(f"-- -- Samples from both classes are required for "
        #                  "retraining the model")
        #     return
        #
        # df_validation = self.__sample_validation_data()
        #
        # train_data = CustomDatasetMLP(df_train)
        # train_iterator = data.DataLoader(train_data,
        #                                  shuffle=True,
        #                                  batch_size=8)
        # validation_data = CustomDatasetMLP(df_validation)
        # validation_iterator = data.DataLoader(validation_data,
        #                                       shuffle=False,
        #                                       batch_size=8)
        #
        # self.model.train_loop(train_iterator, validation_iterator)
        #
        # self.path2transformers.mkdir(exist_ok=True)
        # torch.save(self.model.state_dict(),
        #            self.path2transformers / 'currentModel.pt')
        #
        # self.df_dataset['sample_weight'] = 1.0
        #
        # result, wrong_predictions = self.eval_model()     # prob_pred
        # self.df_dataset['PU_prediction'] = self.df_dataset['prob_pred'] > 0.5
        # self.df_dataset['PU_prob_pred'] = self.df_dataset['prob_pred']
        # self.df_dataset['prediction'] = self.df_dataset['prob_pred']
        #
        # for col in ['PU_score_0', 'PU_score_1', 'prediction', 'prob_pred']:
        #     self.df_dataset[col] = 0
        #
        # return

    def retrain_model(self, freeze_encoder=True, batch_size=8, epochs=3,
                      annotation_gain=10):
        """
        Re-train the classifier model using annotations.

        Parameters
        ----------
        epochs : int, optional (default=3)
            Number of training epochs
        freeze_encoder : bool, optional (default=True)
            If True, the embedding layer is frozen, so that only the
            classification layers is updated. This is useful to use
            precomputed embedings for large datasets.
        batch_size : int, optional (default=8)
            Batch size
        annotation_gain : int or float, optional (default=10)
            Relative value of an annotated sample with respect to a non-
            annotated one.
        """

        # self.load_model()

        # Training data not annotated (i.e., UNUSED in column 'learned')
        is_train = (self.df_dataset.train_test == TRAIN)

        if 'learned' in self.df_dataset:
            is_tr_unused = is_train & (self.df_dataset.learned == UNUSED)
            is_tr_used = is_train & (self.df_dataset.learned == 1)
            is_tr_new = is_train & (self.df_dataset.learned == 0)
        else:
            is_tr_unused = is_train
            is_tr_used = []
            is_tr_new = []

        self.df_dataset.loc[is_tr_unused, "sample_weight"] = 1
        self.df_dataset.loc[is_tr_used, "sample_weight"] = annotation_gain
        self.df_dataset.loc[is_tr_new, "sample_weight"] = annotation_gain

        self.df_dataset.loc[is_train, "labels"] = self.df_dataset.loc[
            is_train, "PUlabels"]
        if 'annotations' in self.df_dataset:
            self.df_dataset.loc[is_tr_used, "labels"] = self.df_dataset.loc[
                is_tr_used, "annotations"]
            self.df_dataset.loc[is_tr_new, "labels"] = self.df_dataset.loc[
                is_tr_new, "annotations"]

        self._train_model(retrain=True)

        # ################
        # Update dataframe

        # Mark new annotations as used
        if 'learned' in self.df_dataset:
            self.df_dataset.loc[self.df_dataset.learned == 0, 'learned'] = 1

        # df_train = self.__sample_train_data(retrain=True)
        # if len(df_train) == 0:
        #     logging.info(f"-- -- Samples from both classes are required for "
        #                  "retraining the model")
        #     return
        # df_validation = self.__sample_validation_data(retrain=True)
        #
        # train_data = CustomDatasetMLP(df_train)
        # train_iterator = data.DataLoader(train_data,
        #                                  shuffle=True,
        #                                  batch_size=8)
        # validation_data = CustomDatasetMLP(df_validation)
        # validation_iterator = data.DataLoader(validation_data,
        #                                       shuffle=False,
        #                                       batch_size=8)
        #
        # self.model.train_loop(train_iterator, validation_iterator)
        #
        # self.path2transformers.mkdir(exist_ok=True)
        # torch.save(self.model.state_dict(),
        #           self.path2transformers / 'currentModel.pt')
        #
        # result, wrong_predictions = self.eval_model()  # prob_pred
        # self.df_dataset['prediction'] = self.df_dataset['prob_pred'] > 0.5

    def _train_model(self, retrain=False):
        """
        trains the classifier model using annotations

        Returns
        -------
        The loaded model is stored in attribute self.model
        """
        self.load_model()
        df_train = self.__sample_train_data(retrain=retrain)
        if retrain and len(df_train) == 0:
            logging.info(f"-- -- Samples from both classes are required for "
                         "retraining the model")
            return

        df_validation = self.__sample_validation_data(retrain=retrain)

        train_data = CustomDatasetMLP(df_train)
        train_iterator = data.DataLoader(train_data,
                                         shuffle=True,
                                         batch_size=8)
        validation_data = CustomDatasetMLP(df_validation)
        validation_iterator = data.DataLoader(validation_data,
                                              shuffle=False,
                                              batch_size=8)

        self.model.train_loop(train_iterator, validation_iterator)

        self.path2transformers.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(),
                   self.path2transformers / 'currentModel.pt')

        self.df_dataset['sample_weight'] = 1.0
        result, wrong_predictions = self.eval_model()  # prob_pred

        if retrain:
            self.df_dataset['prediction'] = self.df_dataset['prob_pred'] > 0.5
        else:
            self.df_dataset['PU_prediction'] = self.df_dataset['prob_pred'] > 0.5
            self.df_dataset['PU_prob_pred'] = self.df_dataset['prob_pred']
            self.df_dataset['prediction'] = self.df_dataset['prob_pred']
            for col in ['PU_score_0', 'PU_score_1', 'prediction', 'prob_pred']:
                self.df_dataset[col] = 0

    def load_model(self):
        """
        Loads an existing classification model

        Returns
        -------
        The loaded model is stored in attribute self.model
        """
        self.path2transformers.mkdir(exist_ok=True)
        self.model = MLP(768, 1024, 1)

        try:
            self.model.load_state_dict(
                torch.load(self.path2transformers / 'currentModel.pt'))
        except Exception:
            pass

    def train_test_split(self, max_imbalance=None, nmax=None, train_size=0.5,
                         random_state=None):
        """
        Split dataframe dataset into train an test datasets,

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
        No variables are returned.
        """

        df_train, df_test = model_selection.train_test_split(
            self.df_dataset, train_size=train_size, random_state=random_state,
            shuffle=True, stratify=None)

        # Marc train and test samples in the dataset.
        self.df_dataset['train_test'] = UNUSED
        self.df_dataset.loc[df_train.index, 'train_test'] = TRAIN
        self.df_dataset.loc[df_test.index, 'train_test'] = TEST

    def load_model_config(self):
        """
        Not relevant for MLP Classifier. However it gets called in the super
        class and has to be passed

        Parameters
        ----------
        """
        pass

    def eval_model(self, samples="train_test", tag="", batch_size=8):
        """
        # inference
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
        batch_size : int, optiona (default=8)
            Batch size
        """

        self.model.eval()
        df_eval = self.df_dataset[(self.df_dataset.train_test == TRAIN)
                                  | (self.df_dataset.train_test == TEST)]

        # new
        eval_data = CustomDatasetMLP(df_eval)
        eval_iterator = data.DataLoader(
            eval_data, shuffle=False, batch_size=8)

        scores, total_loss, result = self.model.eval_model(
            eval_iterator, device=self.device, batch_size=batch_size)

        self.df_dataset.loc[df_eval.index, 'prob_pred'] = scores

        wrong_predictions = []

        return result, wrong_predictions

    def inferData(self, dPaths):
        """
        infers the dataset

        Parameters
        ----------
        dPaths :

        """
        self.model.eval()
        dh = DataHandler(dPaths, classifier=self.model)
        dh.run('p_d', 'p', PandasModifier)

