import pandas as pd
import logging

import matplotlib.pyplot as plt

# Local imports

# You might need to update the location of the baseTaskManager class
from .base_taskmanager import baseTaskManager
from .data_manager import DataManager
from .query_manager import QueryManager
from .domain_classifier.preprocessor import CorpusDFProcessor
from .utils import plotter

from simpletransformers.classification import ClassificationModel


class TaskManager(baseTaskManager):
    """
    This class extends the functionality of the baseTaskManager class for a
    specific example application

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

        # Attributes that will be initialized in the base class
        self.path2project = None
        self.path2metadata = None
        self.path2config = None
        self.path2source = None
        self.metadata_fname = None
        self.global_parameters = None
        self.state = None
        self.metadata = None
        self.ready2setup = None
        self.set_logs = None
        self.logger = None

        super().__init__(path2project, path2source, config_fname=config_fname,
                         metadata_fname=metadata_fname, set_logs=set_logs)

        # You should modify this path here to create the dictionary with the
        # default folder structure of the proyect.
        # Names will be assumed to be subfolders of the project folder in
        # self.path2project.
        # This list can be modified within an active project by adding new
        # folders. Every time a new entry is found in this list, a new folder
        # is created automatically.
        self.f_struct = {'labels': 'labels',
                         'output': 'output'}

        # Main paths
        # Path to the folder with the corpus files
        self.path2corpus = None
        # Path to the folder with label files
        self.path2labels_out = self.path2project / self.f_struct['labels']

        # Corpus dataframe
        self.corpus_name = None    # Name of the corpus
        self.df_corpus = None      # Corpus dataframe
        self.df_labels = None      # Labels
        self.keywords = None
        self.CorpusProc = None

        # Datamanager
        self.DM = DataManager(self.path2source, self.path2labels_out)

        # Query manager
        self.QM = QueryManager()

        return

    def _get_corpus_list(self):
        """
        Returns the list of available corpus
        """

        return self.DM.get_corpus_list()

    def _get_labelset_list(self):
        """
        Returns the list of available corpus
        """

        if self.corpus_name is None:
            logging.warning("\n")
            logging.warning(
                "-- No corpus loaded. You must load a corpus first")
            labelset_list = []
        else:
            labelset_list = self.DM.get_labelset_list(self.corpus_name)

        return labelset_list

    def _ask_keywords(self):
        """
        Ask the user for a list of keywords.

        Returns
        -------
        keywords : list of str
            List of keywords
        """

        # Read available list of AI keywords
        kw_library = self.DM.get_keywords_list(self.corpus_name)
        # Ask keywords through the query manager
        keywords = self.QM.ask_keywords(kw_library)

        return keywords

    def _ask_label_tag(self):
        """
        Ask the user for a tag to compose the label file name.

        Returns
        -------
        tag : str
            User-defined tag
        """

        return self.QM.ask_label_tag()

    def _ask_topics(self, topic_words):
        """
        Ask the user for a weighted list of topics
        """

        return self.QM.ask_topics(topic_words)

    def load_corpus(self, corpus_name):
        """
        Loads a dataframe of documents from a given corpus.

        Parameters
        ----------
        corpus_name : str
            Name of the corpus. It should be the name of a folder in
            self.path2source
        """

        # Store the name of the corpus an object attribute because later
        # tasks will be referred to this corpus
        self.corpus_name = corpus_name

        # Load corpus in a dataframe.
        self.df_corpus = self.DM.load_corpus(self.corpus_name)
        self.CorpusProc = CorpusDFProcessor(self.df_corpus)

        return

    def import_labels(self):
        """
        Import labels from file
        """

        ids_corpus = self.df_corpus['id']
        self.df_labels, msg = self.DM.import_labels(
            corpus_name=self.corpus_name, ids_corpus=ids_corpus)

        # self.df_labels and msg are returned for the GUI
        return self.df_labels, msg

    def analyze_keywords(self):
        """
        Get a set of positive labels using keyword-based search
        """

        # Get weight parameter (weight of title word wrt description words)
        wt = self.QM.ask_value(
            query=("Introduce the (integer) weight of the title words with "
                   "respect to the description words "),
            convert_to=int,
            default=self.global_parameters['keywords']['wt'])

        if self.keywords is None:
            logging.info("-- No active keywords in this session.")
            self.keywords = self._ask_keywords()

        else:
            logging.info("-- Analyzing current list of active keywords")

        # Weight of the title words
        logging.info(f'-- Selected keywords: {self.keywords}')

        df_stats, kf_stats = self.CorpusProc.compute_keyword_stats(
            self.keywords, wt)
        plotter.plot_top_values(
            df_stats, title="Document frequencies", xlabel="No. of docs")
        plotter.plot_top_values(
            kf_stats, title="Keyword frequencies", xlabel="No. of keywords")

        y = self.CorpusProc.score_by_keywords(self.keywords, wt=20)

        # Plot sorted document scores
        plt.figure()
        plt.plot(sorted(y))
        plt.xlabel('Document')
        plt.ylabel('Score')
        plt.show(block=False)

        return

    def get_labels_by_keywords(self):
        """
        Get a set of positive labels using keyword-based search
        """

        # ##############
        # Get parameters

        # Get weight parameter (weight of title word wrt description words)
        wt = self.QM.ask_value(
            query="Set the (integer) weight of the title words with "
                  "respect to the description words",
            convert_to=int,
            default=self.global_parameters['keywords']['wt'])

        # Get weight parameter (weight of title word wrt description words)
        n_max = self.QM.ask_value(
            query=("Set maximum number of returned documents"),
            convert_to=int,
            default=self.global_parameters['keywords']['n_max'])

        # Get score threshold
        s_min = self.QM.ask_value(
            query=("Set score_threshold"),
            convert_to=float,
            default=self.global_parameters['keywords']['s_min'])

        # #######################
        # Get keywords and labels
        self.keywords = self._ask_keywords()
        tag = self._ask_label_tag()

        logging.info(f'-- Selected keywords: {self.keywords}')

        # Find the documents with the highest scores given the keywords
        ids = self.CorpusProc.filter_by_keywords(
            self.keywords, wt=wt, n_max=n_max, s_min=s_min)

        # Create dataframe of positive labels from the list of ids
        self.df_labels = self.CorpusProc.make_pos_labels_df(ids)

        # ############
        # Save labels
        msg = self.DM.save_labels(self.df_labels, corpus_name=self.corpus_name,
                                  tag=tag)

        return msg

    def get_labels_by_topics(self):
        """
        Get a set of positive labels from a weighted list of topics
        """

        # ##############
        # Get parameters

        # Get weight parameter (weight of title word wrt description words)
        n_max = self.QM.ask_value(
            query=("Introduce maximum number of returned documents"),
            convert_to=int,
            default=self.global_parameters['topics']['n_max'])

        # Get score threshold
        s_min = self.QM.ask_value(
            query=("Introduce score_threshold"),
            convert_to=float,
            default=self.global_parameters['topics']['s_min'])

        # ############################
        # Get topic weights and labels

        # Load topics
        T, df_metadata, topic_words = self.DM.load_topics()

        # Remove all documents (rows) from the topic matrix, that are not
        # in self.df_corpus.
        T, df_metadata = self.CorpusProc.remove_docs_from_topics(
            T, df_metadata, col_id='corpusid')

        # Ask for topic weights
        topic_weights = self._ask_topics(topic_words)
        # Ask tag for the label file
        tag = self._ask_label_tag()

        # Filter documents by topics
        ids = self.CorpusProc.filter_by_topics(
            T, df_metadata['corpusid'], topic_weights, n_max=n_max,
            s_min=s_min)

        # Create dataframe of positive labels from the list of ids
        self.df_labels = self.CorpusProc.make_pos_labels_df(ids)

        # ###########
        # Save labels
        msg = self.DM.save_labels(self.df_labels, corpus_name=self.corpus_name,
                                  tag=tag)

        return msg

    def get_labels_by_definitions(self):

        labels = ['lab1', 'lab2', 'lab3']

        return labels

    def load_labels(self, labelset):

        print(f"-- -- Labelset {labelset} loaded")

        return

    def reset_labels(self, labelset):
        """
        Reset a given set of labels
        """

        path2labelset = self.path2labels_out / labelset
        path2labelset.unlink()
        print(f"-- -- Labelset {labelset} removed")

        return

    def train_model(self):

        prefix = '../yelp_review_polarity_csv/'

        train_df = pd.read_csv(prefix + 'train.csv', header=None)
        train_df.head()

        eval_df = pd.read_csv(prefix + 'test.csv', header=None)
        eval_df.head()

        train_df[0] = (train_df[0] == 2).astype(int)
        eval_df[0] = (eval_df[0] == 2).astype(int)

        train_df = pd.DataFrame({
            'text': train_df[1].replace(r'\n', ' ', regex=True),
            'label': train_df[0]})

        print(train_df.head())

        eval_df = pd.DataFrame({
            'text': eval_df[1].replace(r'\n', ' ', regex=True),
            'label': eval_df[0]})

        print(eval_df.head())

        # ##############
        # Classification

        logging.warning("THE FOLLOWING CODE IS UNDER CONSTRUCTION. BE AWARE "
                        "THAT IT ADDS LARGE FILES INTO THE CODE FOLDERS, THAT "
                        "PRODUCES ERRORS WHEN USING GIT")
        breakpoint()

        # Create a TransformerModel
        model = ClassificationModel('roberta', 'roberta-base', use_cuda=False)

        # Train the model
        model.train_model(train_df)

        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(eval_df)

        a = 1

        return

    def get_feedback(self, image):

        return

    def update_model(self, param):

        return


class TaskManagerGUI(TaskManager):
    """
    Provides extra functionality to the task manager, to be used by the
    Graphical User Interface (GUI)
    """

    def get_suggested_keywords_gui(self):
        """
        Get the list of suggested keywords to showing it in the GUI.

        Returns
        -------
        suggested_keywords : list of str
            List of suggested keywords
        """

        # Read available list of AI keywords
        kw_library = self.DM.get_keywords_list(self.corpus_name)
        suggested_keywords = ', '.join(kw_library)
        logging.info(
            f"-- Suggested list of keywords: {', '.join(kw_library)}\n")

        return suggested_keywords

    def analyze_keywords_gui(self):
        # Weight of the title words
        wt = 2
        logging.info(f'-- Selected keywords: {self.keywords}')

        df_stats, kf_stats = self.CorpusProc.compute_keyword_stats(
            self.keywords, wt)
        plotter.plot_top_values(
            df_stats, title="Document frequencies", xlabel="No. of docs")
        plotter.plot_top_values(
            kf_stats, title="Keyword frequencies", xlabel="No. of keywords")

        y = self.CorpusProc.score_by_keywords(self.keywords, wt=20)

        return y, df_stats, kf_stats

    def get_labels_by_keywords_gui(self, keywords, _tag):
        """
        Get a set of positive labels using keyword-based search through the
        MainWindow
        """

        # Weight of the title words
        wt = 2
        n_max = 2000
        s_min = 1

        self.keywords = keywords
        tag = _tag

        logging.info(f'-- Selected keywords: {self.keywords}')

        # Find the documents with the highest scores given the keywords
        ids = self.CorpusProc.filter_by_keywords(
            self.keywords, wt=wt, n_max=n_max, s_min=s_min)

        # Create dataframe of positive labels from the list of ids
        self.df_labels = self.CorpusProc.make_pos_labels_df(ids)

        # Save labels
        message_out = self.DM.save_labels(
            self.df_labels, corpus_name=self.corpus_name, tag=tag)

        return message_out

    def get_topic_words_gui(self, n_max, s_min):

        # Load topics
        T, df_metadata, topic_words = self.DM.load_topics()

        # Remove all documents (rows) from the topic matrix, that are not
        # in self.df_corpus.
        T, df_metadata = self.CorpusProc.remove_docs_from_topics(
            T, df_metadata, col_id='corpusid')

        return topic_words, T, df_metadata

    def get_labels_by_topics_gui(
            self, topic_weights, tag, T, df_metadata, n_max, s_min):

        # Filter documents by topics
        ids = self.CorpusProc.filter_by_topics(
            T, df_metadata['corpusid'], topic_weights, n_max=n_max,
            s_min=s_min)

        # Create dataframe of positive labels from the list of ids
        self.df_labels = self.CorpusProc.make_pos_labels_df(ids)

        # Save labels
        message_out = self.DM.save_labels(
            self.df_labels, corpus_name=self.corpus_name, tag=tag)

        return message_out

