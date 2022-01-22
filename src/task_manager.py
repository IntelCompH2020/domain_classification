import logging

# Local imports
# You might need to update the location of the baseTaskManager class
from .base_taskmanager import baseTaskManager
from .data_manager import DataManager
from .query_manager import QueryManager
from .domain_classifier.preprocessor import CorpusDFProcessor
from .domain_classifier.classifier import CorpusClassifier
from .utils import plotter


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
                         'transformers': 'transformers',
                         'output': 'output'}

        # Main paths
        # Path to the folder with the corpus files
        self.path2corpus = None
        # Path to the folder with label files
        self.path2labels_out = self.path2project / self.f_struct['labels']
        self.path2transformers = (
            self.path2project / self.f_struct['transformers'])

        # Corpus dataframe
        self.corpus_name = None    # Name of the corpus
        self.df_corpus = None      # Corpus dataframe
        self.df_labels = None      # Labels
        self.keywords = None
        self.CorpusProc = None

        # Classifier results:
        self.result = None
        self.model_outputs = None

        # Datamanager
        self.DM = DataManager(self.path2source, self.path2labels_out)

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

        return msg

    def analyze_keywords(self, wt=2):
        """
        Get a set of positive labels using keyword-based search

        Parameters:
        -----------
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
        """

        # Weight of the title words
        logging.info(f'-- Selected keywords: {self.keywords}')

        df_stats, kf_stats = self.CorpusProc.compute_keyword_stats(
            self.keywords, wt)
        plotter.plot_top_values(
            df_stats, title="Document frequencies", xlabel="No. of docs")
        plotter.plot_top_values(
            kf_stats, title="Keyword frequencies", xlabel="No. of keywords")

        y = self.CorpusProc.score_by_keywords(self.keywords, wt=wt)

        # Plot sorted document scores
        plotter.plot_doc_scores(y)

        return y, df_stats, kf_stats

    def get_labels_by_keywords(self, wt=2, n_max=2000, s_min=1, tag="kwds"):
        """
        Get a set of positive labels using keyword-based search

        Parameters:
        -----------
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
        n_max: int or None, optional (defaul=2000)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=1)
            Minimum score. Only elements strictly above s_min are selected
        tag: str, optional (default=1)
            Name of the output label set.
        """

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

    def get_labels_by_topics(self, topic_weights, T, df_metadata, n_max=2000,
                             s_min=1, tag="tpcs"):
        """
        Get a set of positive labels from a weighted list of topics

        Parameters:
        -----------
        topic_weights: numpy.array
            Weight of each topic
        T: numpy.ndarray
            Topic matrix
        df_metadata:
            Topic metadata
        n_max: int or None, optional (defaul=2000)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=1)
            Minimum score. Only elements strictly above s_min are selected
        tag: str, optional (default=1)
            Name of the output label set.
        """

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
        """
        Train a domain classifiers
        """

        logging.info("-- Loading PU dataset")
        df_dataset = self.CorpusProc.make_PU_dataset(self.df_labels)

        dc = CorpusClassifier(path2transformers=self.path2transformers)
        max_imbalance = 3
        nmax = 400

        df_train, df_test = dc.train_test_split(
            df_dataset, max_imbalance=max_imbalance, nmax=nmax)

        self.result, self.model_outputs, wrong_predictions = dc.train_model(
            df_train, df_test)

        print("Stopping after training. You should go step by step here")
        input("Press enter...")

        # Pretty print dictionary of results
        logging.info(f"-- Classification results: {self.result}")
        for r, v in self.result.items():
            logging.info(f"-- -- {r}: {v}")

        return

    def get_feedback(self, image):
        """
        """

        a = 1

        return a

    def update_model(self, param):

        return


class TaskManagerCMD(TaskManager):
    """
    Provides extra functionality to the task manager, requesting parameters
    from users from a command window.
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
        """

        super().__init__(path2project, path2source, config_fname=config_fname,
                         metadata_fname=metadata_fname, set_logs=set_logs)

        # Query manager
        self.QM = QueryManager()

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

        y, df_stats, kf_stats = super().analyze_keywords(wt)

        return y, df_stats, kf_stats

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

        # Get keywords and labels
        self.keywords = self._ask_keywords()
        tag = self._ask_label_tag()

        # ##########
        # Get labels
        msg = super().get_labels_by_keywords(
            wt=wt, n_max=n_max, s_min=s_min, tag=tag)

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

        # #################
        # Get topic weights

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

        # ##########
        # Get labels
        msg = super().get_labels_by_topics(topic_weights, T, df_metadata,
                                           n_max=n_max, s_min=s_min, tag=tag)

        return msg


class TaskManagerGUI(TaskManager):
    """
    Provides extra functionality to the task manager, to be used by the
    Graphical User Interface (GUI)
    """

    def get_suggested_keywords(self):
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

    def get_labels_by_keywords(self, keywords, _tag):
        """
        Get a set of positive labels using keyword-based search through the
        MainWindow
        """

        # Keywords are received as arguments
        self.keywords = keywords

        msg = super().get_labels_by_keywords(wt=2, n_max=2000, s_min=1,
                                             tag=_tag)

        return msg

    def get_topic_words(self, n_max, s_min):

        # Load topics
        T, df_metadata, topic_words = self.DM.load_topics()

        # Remove all documents (rows) from the topic matrix, that are not
        # in self.df_corpus.
        T, df_metadata = self.CorpusProc.remove_docs_from_topics(
            T, df_metadata, col_id='corpusid')

        return topic_words, T, df_metadata
