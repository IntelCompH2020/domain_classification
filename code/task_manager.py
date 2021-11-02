import numpy as np
import pandas as pd
import scipy.sparse as scp
import logging
import pathlib

# Local imports

# You might need to update the location of the baseTaskManager class
from .base_taskmanager import baseTaskManager


class TaskManager(baseTaskManager):
    """
    This is an example of task manager that extends the functionality of
    the baseTaskManager class for a specific example application

    This class inherits from the baseTaskManager class, which provides the
    basic method to create, load and setup an application project.

    The behavior of this class might depend on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    def __init__(self, path2project, path2source=None,
                 config_fname='parameters.yaml', metadata_fname='metadata.pkl',
                 set_logs=True):
        """
        Opens a task manager object.

        Parameters
        ----------
        path2project : pathlib.Path
            Path to the application project
        path2source : pathlib.Path
            Path to the folder containing the data sources
        config_fname : str, optional (default='parameters.yaml')
            Name of the configuration file
        metadata_fname : str or None, optional (default=None)
            Name of the project metadata file.
            If None, no metadata file is used.
        set_logs : bool, optional (default=True)
            If True logger objects are created according to the parameters
            specified in the configuration file
        path2source : pathlib.Path
            Path to the source data.
        """

        super().__init__(path2project, path2source, config_fname=config_fname,
                         metadata_fname=metadata_fname, set_logs=set_logs)

        # You should modify this path here to create the dictionary with the
        # default folder structure of the proyect.
        # Names will be assumed to be subfolders of the project folder in
        # self.path2project.
        # This list can be modified within an active project by adding new
        # folders. Every time a new entry is found in this list, a new folder
        # is created automatically.
        self.f_struct = {'figures': 'figures',
                         'output': 'output'}

        # Corpus dataframe
        self.corpus_name = None    # Name of the corpus
        self.path2corpus = None    # Path to the folder with the corpus files
        self.df_corpus = None      # Corpus dataframe
        self.df_labels = None      # Labels

        return

    def _get_corpus_list(self):
        """
        Returns the list of available corpus
        """

        corpus_list = [e.name for e in self.path2source.iterdir()
                       if e.is_dir()]

        return corpus_list

    def _get_labels_list(self):
        """
        Returns the list of available corpus
        """
        corpus_list = ['arxiv']

        return corpus_list

    def load_corpus(self, corpus_name):
        """
        Loads a dataframe of documents from a given corpus.

        Parameters
        ----------
        corpus_name : str
            Name of the corpus. It should be the name of a folder in
            self.path2source
        """

        self.corpus_name = corpus_name
        self.path2corpus = self.path2source / self.corpus_name

        # ####################
        # Paths and file names

        # Some file and folder names that could be specific of the current
        # corpus. They should be possibly moved to a config file
        corpus_fpath1 = pathlib.Path('corpus') / 'Cordis-fp7' / 'project.xlsx'
        corpus_fpath2 = (pathlib.Path('corpus') / 'Cordis-H2020'
                         / 'project.xlsx')

        # ###########
        # Load corpus

        # Load corpus 1
        path2corpus1 = self.path2corpus / corpus_fpath1
        df_corpus1 = pd.read_excel(path2corpus1)
        logging.info(f'-- Corpus fp7 loaded with {len(df_corpus1)} documents')
        path2corpus2 = self.path2corpus / corpus_fpath2
        df_corpus2 = pd.read_excel(path2corpus2)

        # The following is because df_corpus2['frameworkProgramme'] appears as
        # "H2020;H2020H...;H2020" in some cases
        df_corpus2['frameworkProgramme'] = 'H2020'

        logging.info(
            f'-- Corpus H2020 loaded with {len(df_corpus2)} documents')

        # Test if corpus ids overlap
        doc_ids_in_c1 = list(df_corpus1['id'])
        doc_ids_in_c2 = list(df_corpus2['id'])
        common_ids = set(doc_ids_in_c1).intersection(doc_ids_in_c2)
        if len(common_ids) > 0:
            logging.warn(f"-- There are {len(common_ids)} ids appearing in "
                         'both corpus')

        # Join corpus.
        # Original fields are:
        # 'id', 'acronym', 'status', 'title', 'startDate', 'endDate',
        # 'totalCost', 'ecMaxContribution', 'frameworkProgramme', 'subCall',
        # 'fundingScheme', 'nature', 'objective', 'contentUpdateDate', 'rcn'
        self.df_corpus = df_corpus1.append(df_corpus2)
        breakpoint()
        self.df_corpus = self.df_corpus[['id', 'acronym', 'title',
                                         'objective', 'rcn']]

        return

    def import_labels(self):

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
        self.df_labels = pd.read_excel(labels_fpath)
        # Original label file contains 'id', 'Title', 'Abstract',
        self.df_labels.rename(columns={'Project ID': 'id'}, inplace=True)
        # All samples are (hopefully) from the positive class
        self.df_labels['class'] = 1

        # ############
        # Check labels

        # Check if all labeled docs are in the corpus.
        ids_labels = self.df_labels['id']
        breakpoint()
        ids_corpus = self.df_corpus['rcn']
        strange_labels = set(ids_labels) - set(ids_corpus)
        if len(strange_labels) > 0:
            logging.warn(f"-- There are {len(strange_labels)} documents in the"
                         " labeled dataset that do not belong to the corpus")
        # common_labels = set(ids_labels) - strange_labels

        logging.info(f"-- File with {len(self.df_labels)} labels from the "
                     "positive class loaded")

        return

    def get_labels_by_keywords(self):

        labels = ['lab1', 'lab2', 'lab3']

        return labels

    def get_labels_by_topics(self):

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

        # Extract topic model info:
        T = scp.csr_matrix((data['thetas_data'], data['thetas_indices'],
                            data['thetas_indptr']))
        n_docs, n_topics = T.shape

        logging.info(f'-- Topic matrix loaded with {n_topics} topics and'
                     f' {n_docs} documents')

        # Load metadata
        path2metadata = self.path2corpus / topic_folder / metadata_fname
        df_metadata = pd.read_csv(path2metadata)
        logging.info(f'-- Topic matrix metadata loaded about '
                     f'{len(df_metadata)} documents')
        df_metadata.head()
        docs_in_tm = list(df_metadata['corpusid'])

        print("Up to here.")

        return

    def get_labels_by_definitions(self):

        labels = ['lab1', 'lab2', 'lab3']

        return labels

    def load_labels(self, labelset):

        labels = ['lab1', 'lab2', 'lab3']

        return labels

    def reset_labels(self, labelset):
        """
        Reset a given set of labels
        """

        pass

        return

    def train_models(self):

        images = ['im1', 'im2', 'im3']

        return images

    def get_feedback(self, image):

        return

    def update_model(self, param):

        return
