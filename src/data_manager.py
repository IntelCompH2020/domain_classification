"""
Defines a data manager class to provide basic read-write functionality for
the project.

@author: J. Cid-Sueiro, A. Gallardo-Antolin
"""

import pathlib
import pandas as pd
import numpy as np
import scipy.sparse as scp
import logging
import shutil

from time import time
from sklearn.preprocessing import normalize


class DataManager(object):
    """
    This class contains all read / write functionalities required by the
    domain_classification project.

    It assumes that source and destination data will be stored in files.
    """

    def __init__(self, path2source, path2labels, path2datasets, path2models,
                 path2embeddings=None):
        """
        Initializes the data manager object

        Parameters
        ----------
        path2source: str or pathlib.Path
            Path to the folder containing all external source data
        path2labels: str or pathlib.Path
            Path to the folder containing sets of labels
        path2datasets: str or pathlib.Path
            Path to the folder containing datasets
        path2embeddings: str or pathlib.Path
            Path to the folder containing the document embeddings
        """

        self.path2source = pathlib.Path(path2source)
        self.path2labels = pathlib.Path(path2labels)
        self.path2datasets = pathlib.Path(path2datasets)
        self.path2models = pathlib.Path(path2models)
        if path2embeddings is not None:
            self.path2embeddings = pathlib.Path(path2embeddings)
        # The path to the corpus is given when calling the load_corpus method
        self.path2corpus = None

        # Default name of the corpus. It can be changed by the load_corpus or
        # the load_dataset methods
        self.corpus_name = ""

        return

    def get_corpus_list(self):
        """
        Returns the list of available corpus
        """

        corpus_list = [e.name for e in self.path2source.iterdir()
                       if e.is_dir()]

        return corpus_list

    def get_labelset_list(self):
        """
        Returns the list of available labels
        """

        prefix = f"labels_{self.corpus_name}_"
        labelset_list = [e.stem for e in self.path2labels.iterdir()
                         if e.is_file() and e.stem.startswith(prefix)]
        # Remove prefixes and get tags only:
        labelset_list = [e[len(prefix):] for e in labelset_list]

        return labelset_list

    def get_dataset_list(self):
        """
        Returns the list of available datasets
        """

        prefix = f"dataset_{self.corpus_name}_"
        dataset_list = [e.stem for e in self.path2datasets.iterdir()
                        if e.is_file() and e.stem.startswith(prefix)]
        # Remove prefixes and get tags only:
        dataset_list = [e[len(prefix):] for e in dataset_list]

        # Since the dataset folder can contain csv and feather versions of the
        # same dataset, we remove duplicates
        dataset_list = list(set(dataset_list))

        return dataset_list

    def get_model_list(self):
        """
        Returns the list of available models
        """

        model_list = [e.stem for e in self.path2models.iterdir() if e.is_dir()]

        return model_list

    def get_keywords_list(self, filename='IA_keywords_SEAD_REV_JAG.txt'):
        """
        Returns a list of IA-related keywords read from a file.

        Parameters
        ----------
        filename : str, optional (default=='IA_keywords_SEAD_REV_JAG.txt')
            Name of the file with the keywords

        Returns
        -------
        keywords: list
            A list of keywords (empty if the file does not exist)
        """

        keywords_fpath = (self.path2source / self.corpus_name / 'queries'
                          / filename)

        keywords = []
        if keywords_fpath.is_file():
            df_keywords = pd.read_csv(keywords_fpath, delimiter=',',
                                      names=['keywords'])
            keywords = list(df_keywords['keywords'])

        return keywords

    def load_corpus(self, corpus_name):
        """
        Loads a dataframe of documents from a given corpus.

        Parameters
        ----------
        corpus_name : str
            Name of the corpus. It should be the name of a folder in
            self.path2source
        """

        # Loading corpus
        logging.info(f'-- Loading corpus {corpus_name}')

        self.path2corpus = self.path2source / corpus_name
        path2feather = self.path2corpus / 'corpus' / 'corpus.feather'
        self.corpus_name = corpus_name

        # If there is a feather file, load it
        t0 = time()
        if path2feather.is_file():

            logging.info(f'-- -- Feather file {path2feather} found...')
            df_corpus = pd.read_feather(path2feather)
            logging.info(
                f'-- -- Feather file loaded in {time() - t0:.2f} secs.')
            logging.info(f'-- -- Corpus {corpus_name} loaded with '
                         f'{len(df_corpus)} documents')

        # if it doesn't exist, load the selected corpus in its original form
        elif corpus_name == 'EU_projects':

            # ####################
            # Paths and file names

            # Some file and folder names that could be specific of the current
            # corpus. They should be possibly moved to a config file
            corpus_fpath1 = (pathlib.Path('corpus') / 'Cordis-fp7'
                             / 'project.xlsx')
            corpus_fpath2 = (pathlib.Path('corpus') / 'Cordis-H2020'
                             / 'project.xlsx')

            # ###########
            # Load corpus

            # Load corpus 1
            path2corpus1 = self.path2corpus / corpus_fpath1
            df_corpus1 = pd.read_excel(path2corpus1, engine='openpyxl')
            # Remove duplicates, if any
            df_corpus1.drop_duplicates(inplace=True)
            logging.info(
                f'-- -- Corpus fp7 loaded with {len(df_corpus1)} documents')
            path2corpus2 = self.path2corpus / corpus_fpath2
            df_corpus2 = pd.read_excel(path2corpus2, engine='openpyxl')
            # Remove duplicates, if any
            df_corpus2.drop_duplicates(inplace=True)

            # The following is because df_corpus2['frameworkProgramme']
            # appears as "H2020;H2020H...;H2020" in some cases
            df_corpus2['frameworkProgramme'] = 'H2020'

            logging.info(
                f'-- -- Corpus H2020 loaded with {len(df_corpus2)} documents')

            # Test if corpus ids overlap
            doc_ids_in_c1 = list(df_corpus1['id'])
            doc_ids_in_c2 = list(df_corpus2['id'])

            common_ids = set(doc_ids_in_c1).intersection(doc_ids_in_c2)
            if len(common_ids) > 0:
                logging.warn(f"-- -- There are {len(common_ids)} ids "
                             "appearing in both corpus")

            # Join corpus.
            # Original fields are:
            #     id, acronym, status, title, startDate, endDate, totalCost,
            #     ecMaxContribution, frameworkProgramme, subCall,
            #     fundingScheme, nature, objective, contentUpdateDate, rcn
            # WARNING: We DO NOT use "id" as the project id.
            #          We use "rcn" instead, renamed as "id"
            df_corpus = df_corpus1.append(df_corpus2, ignore_index=True)
            df_corpus = df_corpus[['acronym', 'title', 'objective', 'rcn']]

            # Map column names to normalized names
            mapping = {'rcn': 'id',
                       'objective': 'description'}
            df_corpus.rename(columns=mapping, inplace=True)

            # Fill nan cells with empty strings
            df_corpus.fillna("", inplace=True)

            logging.info(f"-- -- Corpus aggregated with {len(df_corpus)} "
                         f" documents and loaded in {time() - t0:.2f} secs.")
            logging.info(f'-- -- Writing feather file in {path2feather} '
                         f'to speed up future loads')
            df_corpus.to_feather(path2feather)

        elif corpus_name == 'AEI_projects':

            # ####################
            # Paths and file names

            # Some file and folder names that could be specific of the current
            # corpus. They should be possibly moved to a config file
            corpus_fpath = pathlib.Path('corpus') / 'AEI_Public.xlsx'

            # ###########
            # Load corpus

            # Load corpus 1
            path2corpus = self.path2corpus / corpus_fpath
            df_corpus = pd.read_excel(path2corpus, engine='openpyxl')
            # Remove duplicates, if any
            df_corpus.drop_duplicates(subset=['Referencia'], inplace=True)
            logging.info(f'-- -- Raw corpus {corpus_name} loaded with '
                         f'{len(df_corpus)} documents')

            # Remove documents with missing data, if any
            ind_notna = df_corpus['title'].notna()
            df_corpus = df_corpus[ind_notna]
            ind_notna = df_corpus['abstract'] == 0
            df_corpus = df_corpus[~ind_notna]

            # Original fields are:
            #     Año, Convocatoria, Referencia, Área, Subárea, Título,
            #     Palabras Clave, C.I.F., Entidad, CC.AA., Provincia,
            #     € Conced., Resument, title, abstract, keywords,
            #     Ind2017_BIO, Ind2017_TIC, Ind2017_ENE
            df_corpus = df_corpus[[
                'Referencia', 'title', 'abstract', 'Ind2017_BIO',
                'Ind2017_TIC', 'Ind2017_ENE']]

            # Map column names to normalized names
            # We use "Referencia", renamed as "id", as the project id.
            mapping = {'Referencia': 'id',
                       'abstract': 'description',
                       'Ind2017_BIO': 'target_bio',
                       'Ind2017_TIC': 'target_tic',
                       'Ind2017_ENE': 'target_ene'}
            df_corpus.rename(columns=mapping, inplace=True)

            # Fill nan cells with empty strings
            df_corpus.fillna("", inplace=True)

            # Remove special characters
            df_corpus['title'] = df_corpus['title'].str.replace('\t', '')
            df_corpus['description'] = (
                df_corpus['description'].str.replace('\t', ''))

            # Reset the index and drop the old index
            df_corpus = df_corpus.reset_index(drop=True)

            logging.info(
                f"-- -- Corpus {corpus_name} reduced to {len(df_corpus)} "
                f" documents")
            logging.info(f"-- -- Loaded in {time() - t0:.2f} secs.")
            logging.info(f'-- -- Writing feather file in {path2feather} '
                         f'to speed up future loads')
            df_corpus.to_feather(path2feather)

        else:
            logging.warning("-- Unknown corpus")
            df_corpus = None

        return df_corpus

    def load_topics(self):
        """
        Loads a topic matrix for a specific corpus
        """

        # ####################
        # Paths and file names

        # Some file and folder names that could be specific of the current
        # corpus. They should be possibly moved to a config file
        topic_folder = 'topic_model'
        topic_model_fname = 'modelo_sparse.npz'
        metadata_fname = 'CORDIS720all-metadata.csv'

        # ###########
        # Topic model

        # Load topic model
        path2topics = self.path2corpus / topic_folder / topic_model_fname

        if not pathlib.Path.exists(path2topics):
            logging.warning(f"-- No topic model available at {path2topics}")
            return None, None, None

        data = np.load(path2topics)

        # Extract topic descriptions
        topic_words = data['descriptions']

        # Extract topic model info:
        T = scp.csr_matrix((data['thetas_data'], data['thetas_indices'],
                            data['thetas_indptr']))

        # Make sure that topic vectors are normalized
        T = normalize(T, norm='l1', axis=1)

        n_docs, n_topics = T.shape

        logging.info(f'-- Topic matrix loaded with {n_topics} topics and'
                     f' {n_docs} documents')

        # Load metadata
        path2metadata = self.path2corpus / topic_folder / metadata_fname
        df_metadata = pd.read_csv(path2metadata)
        logging.info(f'-- Topic matrix metadata loaded about '
                     f'{len(df_metadata)} documents')

        return T, df_metadata, topic_words

    def load_dataset(self, tag=""):
        """
        Loads a labeled dataset of documents in the format required by the
        classifier modules

        Parameters
        ----------
        tag : str, optional (default="")
            Name of the dataset
        """

        logging.info("-- Loading dataset")

        feather_fname = f'dataset_{self.corpus_name}_{tag}.feather'
        csv_fname = f'dataset_{self.corpus_name}_{tag}.csv'

        # If there is a feather file, load it
        t0 = time()
        path2dataset = self.path2datasets / feather_fname
        if path2dataset.is_file():
            df_dataset = pd.read_feather(path2dataset)
        else:
            # Rename path to load a csv file.
            path2dataset = self.path2datasets / csv_fname
            df_dataset = pd.read_csv(path2dataset)

        msg = (f"-- -- Dataset with {len(df_dataset)} samples loaded from "
               f"{path2dataset} in {time() - t0} secs.")

        logging.info(msg)

        # The log message is returned to be shown in a GUI, if needed
        return df_dataset, msg

    def load_labels(self, tag=""):
        """
        Loads a set or PU labels
        """

        logging.info(f"-- Loading labelset {tag}")

        # Read labels from csv file
        fname = f'labels_{self.corpus_name}_{tag}.csv'
        path2labels = self.path2labels / fname
        df_labels = pd.read_csv(path2labels)

        msg = f"-- -- {len(df_labels)} labels loaded from {path2labels}"
        logging.info(msg)

        # The log message is returned to be shown in a GUI, if needed
        return df_labels, msg

    def reset_labels(self, tag=""):
        """
        Delete all files related to a given class

        Parameters
        ----------
        tag : str, optional (default="")
            Name of the class to be removed
        """

        # Remove csv file
        fname = f"labels_{self.corpus_name}_{tag}.csv"
        path2labelset = self.path2labels / fname
        path2labelset.unlink()

        # Remove dataset
        fstem = f"dataset_{self.corpus_name}_{tag}"
        for p in self.path2datasets.glob(f"{fstem}.*"):
            p.unlink()

        # Remove model
        path2model = self.path2models / tag
        if path2model.is_dir():
            shutil.rmtree(self.path2models / tag)

        logging.info(f"-- -- Labelset {tag} removed")

    def import_labels(self, ids_corpus=None, tag="imported"):
        """
        Loads a subcorpus of positive labels from file.

        Parameters
        ----------
        ids_corpus: list
            List of ids of the documents in the corpus. Only the labels with
            ids in ids_corpus are imported and saved into the output file.
        tag: str, optional (default="imported")
            Name for the category defined by the positive labels.

        Returns
        -------
        df_labels: pandas.DataFrame
            Dataframe of labels, with two columns: id and class.
            id identifies the document corresponding to the label.
            class identifies the class. All documents are assumed to be class 1
        """

        # ####################
        # Paths and file names

        # Some file and folder names that could be specific of the current
        # corpus. They should be possibly moved to a config file
        path2labels = self.path2corpus / 'labels'
        labels_fname = 'CORDIS720_3models.xlsx'

        # ####################
        # Doc selection for AI

        # Load excel file of "models"
        labels_fpath = path2labels / labels_fname
        df_labels = pd.read_excel(labels_fpath)

        # Original label file contains 'Project ID', 'Title', 'Abstract',
        # Normalize column names:
        mapping = {'Project ID': 'id'}
        # Use this mapping just in case the title and descrition are needed:
        # mapping = {'Project ID': 'id', 'Title': 'title',
        #            'Abstract': 'description'}
        df_labels.rename(columns=mapping, inplace=True)

        # Select ids only
        df_labels = df_labels[['id']]
        # Label all samples with the positive class
        df_labels['class'] = 1

        # #############
        # Check samples

        # Check if all labeled docs are in the corpus.
        ids_labels = df_labels['id']
        if ids_corpus is not None:
            strange_labels = set(ids_labels) - set(ids_corpus)
        else:
            strange_labels = []

        if len(strange_labels) > 0:
            logging.warn(
                f"-- Removing {len(strange_labels)} documents from the "
                f"labeled dataset that do not belong to the corpus.")
            df_labels = df_labels[df_labels.id.isin(ids_corpus)]

        # ########################
        # Saving id and class only

        msg = self.save_labels(df_labels, tag=tag)

        # The log message is returned to be shown in a GUI, if needed
        return df_labels, msg

    def save_labels(self, df_labels, tag=""):

        # ########################
        # Saving id and class only

        labels_out_fname = f'labels_{self.corpus_name}_{tag}.csv'
        path2labels = self.path2labels / labels_out_fname
        df_labels.to_csv(path2labels, index=False)

        msg = (f"-- File with {len(df_labels)} positive labels saved in "
               f"{path2labels}")
        logging.info(msg)

        # The log message is returned to be shown in a GUI, if needed
        return msg

    def save_dataset(self, df_dataset, tag="", save_csv=False):
        """
        Save dataset in input dataframe in a feather file.

        Parameters
        ----------
        df_dataset : pandas.DataFrame
            Dataset to save
        tag : str, optional (default="")
            Optional string to add to the output file name.
        save_csv : boolean, optional (default=False)
            If True, the dataset is saved in csv format too
        """

        # ########################
        # Saving id and class only

        dataset_fname = f'dataset_{self.corpus_name}_{tag}.feather'
        path2dataset = self.path2datasets / dataset_fname
        df_dataset.to_feather(path2dataset)
        msg = (f"-- File with {len(df_dataset)} samples saved in "
               f"{path2dataset}")

        if save_csv:
            dataset_fname = f'dataset_{self.corpus_name}_{tag}.csv'
            path2dataset = self.path2datasets / dataset_fname
            df_dataset.to_csv(path2dataset)

        logging.info(msg)

        # The log message is returned to be shown in a GUI, if needed
        return msg

