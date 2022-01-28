import logging
from datetime import datetime

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
                 config_fname='parameters.yaml',
                 metadata_fname='metadata.yaml', set_logs=True):
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
        self.dc = None

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
                         'datasets': 'datasets',
                         'transformers': 'transformers',
                         'output': 'output'}

        # Main paths
        # Path to the folder with the corpus files
        self.path2corpus = None
        # Path to the folder with label files
        self.path2labels = self.path2project / self.f_struct['labels']
        self.path2dataset = self.path2project / self.f_struct['datasets']
        self.path2transformers = (
            self.path2project / self.f_struct['transformers'])

        # Corpus dataframe
        self.df_corpus = None      # Corpus dataframe
        self.df_labels = None      # Labels
        self.keywords = None
        self.CorpusProc = None

        # Classifier results:
        self.df_dataset = None
        self.result = None
        self.model_outputs = None

        # Extend base variables (defined in the base class) for state and
        # metadata with additional fields
        self.state['selected_corpus'] = False  # True if corpus was selected
        self.state['trained_model'] = False    # True if model was trained
        self.metadata['corpus_name'] = None

        # Datamanager
        self.DM = DataManager(
            self.path2source, self.path2labels, self.path2dataset)

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

        # Just to abbreviate
        corpus_name = self.metadata['corpus_name']

        if corpus_name is None:
            logging.warning("\n")
            logging.warning(
                "-- No corpus loaded. You must load a corpus first")
            labelset_list = []
        else:
            labelset_list = self.DM.get_labelset_list(corpus_name)

        return labelset_list

    def _save_dataset(self):
        """
        Saves the dataset.
        The task is done by the self.DM clas. This method is just a caller to
        self.DM, used to simplify (just a bit) the code of methods saving the
        dataset.
        """

        # Update status.
        # Since training takes much time, we store the classification results
        # in files
        self.DM.save_dataset(
            self.df_dataset, corpus_name=self.metadata['corpus_name'],
            save_csv=True)

    def load(self):
        """
        Extends the load method from the parent class to load the project
        corpus and the dataset (if any)
        """

        super().load()
        msg = ""

        # Just to abbreviate
        corpus_name = self.metadata['corpus_name']

        # Restore context from the last execution of the project
        if self.state['selected_corpus']:
            # Load corpus of the project
            self.load_corpus(corpus_name)

            if self.state['trained_model']:
                # Load dataset from the last trained model
                self.df_dataset, msg = self.DM.load_dataset(corpus_name)

                logging.info("-- Loading classification model")
                self.dc = CorpusClassifier(
                    self.df_dataset, path2transformers=self.path2transformers)
                self.dc.load_model()

        return msg

    def load_corpus(self, corpus_name):
        """
        Loads a dataframe of documents from a given corpus.

        Parameters
        ----------
        corpus_name : str
            Name of the corpus. It should be the name of a folder in
            self.path2source
        """

        # The corpus cannot be changed inside the same project. If a corpus
        # was used before we must keep the same one.
        current_corpus = self.metadata['corpus_name']
        if (self.state['selected_corpus'] and corpus_name != current_corpus):
            logging.error(
                f"-- The corpus of this project is {current_corpus}. "
                f"Run another project to use {corpus_name}")
            return

        # Load corpus in a dataframe.
        self.df_corpus = self.DM.load_corpus(corpus_name)
        self.CorpusProc = CorpusDFProcessor(self.df_corpus)

        if not self.state['selected_corpus']:
            # Store the name of the corpus an object attribute because later
            # tasks will be referred to this corpus
            self.metadata['corpus_name'] = corpus_name
            self.state['selected_corpus'] = True
            self._save_metadata()

        return

    def import_labels(self):
        """
        Import labels from file
        """

        ids_corpus = self.df_corpus['id']
        self.df_labels, msg = self.DM.import_labels(
            corpus_name=self.metadata['corpus_name'],
            ids_corpus=ids_corpus)

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
        msg = self.DM.save_labels(
            self.df_labels, corpus_name=self.metadata['corpus_name'], tag=tag)

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
        msg = self.DM.save_labels(
            self.df_labels, corpus_name=self.metadata['corpus_name'], tag=tag)

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

        path2labelset = self.path2labels / labelset
        path2labelset.unlink()
        logging.info(f"-- -- Labelset {labelset} removed")

        return

    def train_PUmodel(self):
        """
        Train a domain classifiers
        """

        logging.info("-- Loading PU dataset")
        self.df_dataset = self.CorpusProc.make_PU_dataset(self.df_labels)

        # Labels from the PU dataset are stored in column "PUlabels". We must
        # copy them to column "labels" which is the name required by
        # simpletransformers
        self.df_dataset[['labels']] = self.df_dataset[['PUlabels']]

        self.dc = CorpusClassifier(
            self.df_dataset, path2transformers=self.path2transformers)

        # Select data for training and testing
        max_imbalance = 3
        nmax = 400
        self.dc.train_test_split(max_imbalance=max_imbalance, nmax=nmax)

        # Train the model using simpletransformers
        self.dc.train_model()

        # Update dataset.
        # This is a bit weird, because the dataset is an attribute of self
        # and self.dc. The following command makes sure they are both equal
        self.df_dataset = self.dc.df_dataset

        # Update status.
        # Since training takes much time, we store the classification results
        # in files
        self._save_dataset()
        self.state['trained_model'] = True
        self._save_metadata()

        return

    def evaluate_PUmodel(self):
        """
        Evaluate a domain classifiers
        """

        # Evaluate the model over the test set
        result, wrong_predictions = self.dc.eval_model(tag_score='PUscore')

        # Pretty print dictionary of results
        logging.info(f"-- Classification results: {result}")
        for r, v in result.items():
            logging.info(f"-- -- {r}: {v}")

        # Update dataset.
        # This is a bit weird, because the dataset is an attribute of self
        # and self.dc. The following command makes sure they are both equal
        self.df_dataset = self.dc.df_dataset
        # Update dataset file to include scores
        self._save_dataset()

        return result

    def get_feedback(self):
        """
        Gets some labels from a user for a selected subset of documents
        """

        # STEP 1: Select bunch of documents at random
        n_docs = 5
        selected_docs = self.dc.sample(n_samples=n_docs)

        # STEP 2: Request labels
        labels = self.get_labels_from_docs(selected_docs)

        # STEP 3: Annotation date
        #

        # STEP 4: Save feedback
        if 'annotations' not in self.df_dataset:
            self.df_dataset[['annotations']] = -1
        self.df_dataset.loc[selected_docs.index, 'annotations'] = labels

        # Add date to the dataframe
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y %H:%M:%S")
        self.df_dataset.loc[selected_docs.index, 'date'] = date_str
        print("now =", now)
        print("date and time =", date_str)
        breakpoint()

        # Update dataset file to include new labels
        self._save_dataset()

        return

    def get_labels_from_docs(self, n_docs):
        """
        Requests feedback about the class of given documents.

        Parameters
        ----------
        selected_docs : pands.DataFrame
            Selected documents

        Returns
        -------
        labels : list of boolean
            Labels for the given documents, in the same order than the
            documents in the input dataframe
        """

        raise NotImplementedError(
            "Please Implement this method in your inherited class")

        return

    def update_model(self):
        """
        Improves classifier performance using the labels provided by users
        """

        # Add updated dataset with the latest annotations
        self.dc.df_dataset = self.df_dataset

        # Retrain model using the new labels
        self.dc.retrain_model()

        return


class TaskManagerCMD(TaskManager):
    """
    Provides extra functionality to the task manager, requesting parameters
    from users from a command window.
    """

    def __init__(self, path2project, path2source=None,
                 config_fname='parameters.yaml',
                 metadata_fname='metadata.yaml', set_logs=True):
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
        kw_library = self.DM.get_keywords_list(self.metadata['corpus_name'])
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

        logging.info(msg)

        return

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

    def get_labels_from_docs(self, selected_docs):
        """
        Requests feedback about the class of given documents.

        Parameters
        ----------
        selected_docs : pands.DataFrame
            Selected documents

        Returns
        -------
        labels : list of boolean
            Labels for the given documents, in the same order than the
            documents in the input dataframe
        """

        print("--Selected docs:")
        print(selected_docs)
        labels = self.QM.ask_labels()

        return labels


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
        kw_library = self.DM.get_keywords_list(self.metadata['corpus_name'])
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

    def get_labels_from_docs(self, selected_docs):
        """
        Requests feedback about the class of given documents.

        Parameters
        ----------
        selected_docs : pands.DataFrame
            Selected documents

        Returns
        -------
        labels : list of boolean
            Labels for the given documents, in the same order than the
            documents in the input dataframe
        """

        # FIXME: Implement this method
        labels = []

        return labels
