import logging
import pathlib
import sys
import os
from time import time

import pandas as pd
import torch
from simpletransformers.classification import ClassificationModel
from sklearn import model_selection


class CorpusClassifier(object):
    """
    A container of corpus classification methods
    """

    def __init__(self, path2transformers="."):
        """
        Initializes a preprocessor object

        Parameters
        ----------
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

        return

    def train_test_split(self, df_dataset, max_imbalance=None, nmax=None,
                         train_size=0.6, random_state=None,
                         class_label='labels'):
        """
        Split dataframe dataset into train an test datasets, undersampling
        the negative class

        Parameters
        ----------
        df_dataset : pandas.DataFrame
            Dataset
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
        class_label: str, optional (default='labels')
            Name of the column containing the class labels.

        Returns
        -------
        df_train: pandas.DataFrame
            Training set
        df_test: pandas.DataFrame
            Test set
        """

        if class_label not in df_dataset:
            logging.error(f"-- -- Column {class_label} does not exist in "
                          "the input dataframe")
            sys.exit()

        l1 = sum(df_dataset[class_label])
        l0 = len(df_dataset) - l1

        # Class balancing
        if (max_imbalance is not None and l0 > max_imbalance * l1):
            # Separate classes
            df0 = df_dataset[df_dataset[class_label] == 0]
            df1 = df_dataset[df_dataset[class_label] == 1]
            # Undersample majority class
            df0 = df0.sample(n=max_imbalance * l1)
            # Re-join dataframes
            df_dataset = pd.concat((df0, df1))

        # Undersampling
        if nmax is not None and nmax < l0 + l1:
            df_dataset = df_dataset.sample(n=nmax)

        df_train, df_test = model_selection.train_test_split(
            df_dataset, train_size=train_size, random_state=random_state,
            shuffle=True, stratify=None)

        return df_train, df_test

    def train_model(self, df_train, df_test, toy_example=False):
        """
        Train and evaluate binary text classification model based on
        transformers

        Parameters
        ----------
        df_train: pandas.DataFrame
            Training set. It must contain columns 'text' and 'labels'
        df_test: pandas.DataFrame
            Test set. It must contain columns 'text' and 'labels'

        Notes
        -----
        The use of simpletransformers follows the example code in
        https://towardsdatascience.com/simple-transformers-introducing-the-
        easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
        """

        # ################
        # Loading toy data

        if toy_example:

            prefix = '../yelp_review_polarity_csv/'
            df_train = pd.read_csv(prefix + 'train.csv', header=None)
            df_test = pd.read_csv(prefix + 'test.csv', header=None)
            # Convert to integers
            df_train[0] = (df_train[0] == 2).astype(int)
            df_test[0] = (df_test[0] == 2).astype(int)

            # Rename headers and remove carriage returns
            df_train = pd.DataFrame({
                'text': df_train[1].replace(r'\n', ' ', regex=True),
                'labels': df_train[0]})
            df_test = pd.DataFrame({
                'text': df_test[1].replace(r'\n', ' ', regex=True),
                'labels': df_test[0]})

            # Reduce datasets for a quick test...
            df_train = df_train.iloc[:1000]
            df_test = df_test.iloc[:100]

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
        model = ClassificationModel(
            'roberta', 'roberta-base', use_cuda=cuda_available,
            args=model_args)

        # Train the model
        logging.info(f"-- -- Training model with {len(df_train)} documents...")
        t0 = time()
        model.train_model(df_train)
        breakpoint()
        logging.info(f"-- -- Model trained in {time() - t0} seconds")

        # Evaluate the model
        logging.info(f"-- -- Testing model with {len(df_test)} documents...")
        t0 = time()
        result, model_outputs, wrong_predictions = model.eval_model(df_test)
        logging.info(f"-- -- Model tested in {time() - t0} seconds")

        return result, model_outputs, wrong_predictions
