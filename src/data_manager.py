import pathlib
import pandas as pd
import numpy as np
import scipy.sparse as scp
import logging
from time import time
from sklearn.preprocessing import normalize


class DataManager(object):
    """
    This class contains all read / write functionalities required by the
    domain_classification project.

    It assumes that source and destination data will be stored in files.
    """

    def __init__(self, path2source, path2labels_out):
        """
        Initializes the data manager object

        Parameters
        ----------
        path2source: str or pathlib.Path
            Path to the folder containing all external source data
        path2labels_out: str or pathlib.Pahd
            Path to the folder containing sets of labels
        """

        self.path2source = pathlib.Path(path2source)
        self.path2labels_out = pathlib.Path(path2labels_out)
        # The path to the corpus is given when calling the load_corpus method
        self.path2corpus = None

        return

    def get_corpus_list(self):
        """
        Returns the list of available corpus
        """

        corpus_list = [e.name for e in self.path2source.iterdir()
                       if e.is_dir()]

        return corpus_list

    def get_labelset_list(self, corpus_name=None):
        """
        Returns the list of available labels

        Parameters
        ----------
        corpus_name: str or None, optional (default=None)
            Name of the corpus whose labels are requested. If None, all
            labelsets are returned
        """

        if corpus_name is None:
            labelset_list = [e.name for e in self.path2labels_out.iterdir()
                             if e.is_file()]
        else:
            labelset_list = [e.name for e in self.path2labels_out.iterdir()
                             if e.is_file() and e.stem.startswith(
                                 f"labels_{corpus_name}")]

        return labelset_list

    def get_keywords_list(self, corpus_name):
        """
        Returns a list of IA-related keywords

        Parameters
        ----------
        corpus_name: str or None
            Name of the corpus whose labels are requested.

        Returns
        -------
        keywords: list
            A list of keywords
        """

        keywords_fpath = (self.path2source / corpus_name / 'queries'
                          / 'IA_keywords_SEAD_REV_JAG.txt')

        df_keywords = pd.read_csv(keywords_fpath)
        keywords = list(df_keywords['artificial neural network'])

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

        # If there is a feather file, load it
        if path2feather.is_file():

            logging.info(f'-- -- Feather file {path2feather} found...')
            t0 = time()
            df_corpus = pd.read_feather(path2feather)
            logging.info(f'-- -- Feather file loaded in {time() - t0} secs.')
            logging.info(f'-- -- Corpus {corpus_name} loaded with '
                         f'{len(df_corpus)} documents')

        # if it doesn't exist, load the selected corpus in its original form
        elif corpus_name == 'EU_projects':

            t0 = time()

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

            logging.info(f"-- -- Aggregated corpus with {len(df_corpus)} "
                         " documents")
            logging.info(f'-- -- Corpus loaded in {time() - t0} secs.')
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

    def import_labels(self, corpus_name="", ids_corpus=None):
        """
        Loads a subcorpus of positive labels from file.

        Parameters
        ----------
        corpus_name: str, optional (default="")
            Name of the corpus. This is used to compose the name of the output
            file of labels only.
        ids_corpus: list
            List of ids of the documents in the corpus. Only the labels with
            ids in ids_corpus are imported and saved into the output file.

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
                f"-- Removing {len(strange_labels)} documents in the labeled "
                f"dataset that do not belong to the corpus.")
            df_labels = df_labels[df_labels.id.isin(ids_corpus)]

        # ########################
        # Saving id and class only

        tag = "imported"
        msg = self.save_labels(df_labels, corpus_name=corpus_name, tag=tag)

        # The log message is returned to be shown in a GUI, if needed
        return df_labels, msg

    def save_labels(self, df_labels, corpus_name="", tag=""):

        # ########################
        # Saving id and class only

        labels_out_fname = f'labels_{corpus_name}_{tag}.csv'
        path2labels_out = self.path2labels_out / labels_out_fname
        df_labels.to_csv(path2labels_out, index=False)

        msg = (f"-- File with {len(df_labels)} positive labels imported and "
               f"saved in {path2labels_out}")
        logging.info(msg)

        # The log message is returned to be shown in a GUI, if needed
        return msg
