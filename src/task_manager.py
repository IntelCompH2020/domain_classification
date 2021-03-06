"""
Defines classes that define methods to run the main tasks in the project,
using the core processing classes and methods.

@author: J. Cid-Sueiro, L. Calvo-Bartolome, A. Gallardo-Antolin
"""

import logging

# Local imports
# You might need to update the location of the baseTaskManager class
from .base_taskmanager import baseTaskManager
from .data_manager import DataManager
from .query_manager import QueryManager
from .domain_classifier.preprocessor import CorpusDFProcessor
from .domain_classifier.classifier import CorpusClassifier
from .utils import plotter

import numpy as np
import matplotlib.pyplot as plt

# A message that is used twice in different parts of the code. It is defined
# here because the same message must be used in both cases.
NO_GOLD_STANDARD = 'Do not use a Gold Standard.'


class TaskManager(baseTaskManager):

    """
    This class extends the functionality of the baseTaskManager class for a
    specific example application

    This class inherits from the baseTaskManager class, which provides the
    basic method to create, load and setup an application project.

    The behavior of this class might depend on the state of the project, in
    dictionary self.state, with the following entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    def __init__(self, path2project, path2source=None, path2zeroshot=None,
                 config_fname='parameters.yaml',
                 metadata_fname='metadata.yaml', set_logs=True):
        """
        Opens a task manager object.

        Parameters
        ----------
        path2project : pathlib.Path
            Path to the application project
        path2source : str or pathlib.Path or None (default=None)
            Path to the folder containing the data sources
        path2zeroshot : str or pathlib.Path or None (default=None)
            Path to the folder containing the zero-shot-model
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
        self.f_struct = {'datasets': 'datasets',
                         'models': 'models',
                         'output': 'output',
                         'embeddings': 'embeddings',
                         # 'labels': 'labels',     # No longer used
                         }

        # Main paths
        # Path to the folder with the corpus files
        self.path2corpus = None
        # Path to the folder with label files
        self.path2datasets = self.path2project / self.f_struct['datasets']
        self.path2models = self.path2project / self.f_struct['models']
        self.path2embeddings = self.path2project / self.f_struct['embeddings']
        self.path2output = self.path2project / self.f_struct['output']

        # Path to the folder containing the zero-shot model
        self.path2zeroshot = path2zeroshot

        # Corpus dataframe
        self.df_corpus = None      # Corpus dataframe
        self.df_dataset = None     # Datasets of inputs, labels and outputs
        self.class_name = None     # Name of the working category
        self.keywords = None
        self.CorpusProc = None

        # Extend base variables (defined in the base class) for state and
        # metadata with additional fields
        self.state['selected_corpus'] = False  # True if corpus was selected
        self.metadata['corpus_name'] = None

        # Datamanager
        self.DM = DataManager(self.path2source, self.path2datasets,
                              self.path2models, self.path2embeddings)

        return

    def _is_model(self, verbose=True):
        """
        Check if labels have been loaded and a domain classifier object has
        been created.
        """

        if self.df_dataset is None:
            if verbose:
                logging.warning("-- No labels loaded. You must load or create "
                                "a set of labels first")
            return False

        elif not self.DM.is_model(self.class_name):
            if verbose:
                logging.warning(
                    f"-- No model exists for class {self.class_name}. You "
                    f"must train a model first")
            return False

        else:
            return True

    def _get_corpus_list(self):
        """
        Returns the list of available corpus
        """

        return self.DM.get_corpus_list()

    def _get_dataset_list(self):
        """
        Returns the list of available corpus
        """

        # Just to abbreviate
        corpus_name = self.metadata['corpus_name']

        if corpus_name is None:
            logging.warning("\n")
            logging.warning(
                "-- No corpus loaded. You must load a corpus first")
            dataset_list = []
        else:
            dataset_list = self.DM.get_dataset_list()

        return dataset_list

    def _get_gold_standard_labels(self):
        """
        Returns the list of gold-standard labels or labelsets available
        in the current corpus.

        Gold-standard labels are those whose name starts with 'target_'

        Gold-standard labels will be used for evaluation only, not for
        learning.
        """

        gs_labels = [x for x in self.df_corpus.columns
                     if x.startswith('target_')]

        if gs_labels == []:
            logging.warning('No Gold Standard is available. Please choose the '
                            'unique option in the menu')
            gs_labels = [NO_GOLD_STANDARD]

        return gs_labels

    def _save_dataset(self):
        """
        Saves the dataset used by the last classifier object.

        Note that this method saves self.dc.df_dataset, not self.df_dataset

        self.dc.df_dataset contains results of the classifier training, as
        well as annotations, that are missed in self.df_dataset

        The task is done by the self.DM clas. This method is just a caller to
        self.DM, used to simplify (just a bit) the code of methods saving the
        dataset.
        """

        # Update status.
        # Since training takes much time, we store the classification results
        # in files
        self.DM.save_dataset(
            self.dc.df_dataset, tag=self.class_name, save_csv=True)

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

        return msg

    def setup(self):
        """
        Sets up the application projetc. To do so, it loads the configuration
        file and activates the logger objects.
        """

        super().setup()

        # Fill global parameters.
        # This is for backward compatibility with project created before the
        # release of this version
        params = {'sampler': 'extremes',
                  'p_ratio': 0.8,
                  'top_prob': 0.1}
        for param, value in params.items():
            if param not in self.global_parameters['active_learning']:
                self.global_parameters['active_learning'][param] = value

        return

    def load_corpus(self, corpus_name):
        """
        Loads a dataframe of documents from a given corpus.

        Parameters
        ----------
        corpus_name : str
            Name of the corpus. It should be the name of a folder in
            self.path2source
        """

        # Dictionary of sampling factor for the corpus loader.
        sampling_factors = self.global_parameters['corpus']['sampling_factor']
        # Default sampling factor: 1 (loads the whole corpus)
        sampling_factor = 1
        if corpus_name in sampling_factors:
            sampling_factor = sampling_factors[corpus_name]

        # The corpus cannot be changed inside the same project. If a corpus
        # was used before we must keep the same one.
        current_corpus = self.metadata['corpus_name']
        if (self.state['selected_corpus'] and corpus_name != current_corpus):
            logging.error(
                f"-- The corpus of this project is {current_corpus}. "
                f"Run another project to use {corpus_name}")
            return

        # Load corpus in a dataframe.
        self.df_corpus = self.DM.load_corpus(
            corpus_name, sampling_factor=sampling_factor)

        self.CorpusProc = CorpusDFProcessor(
            self.df_corpus, self.path2embeddings, self.path2zeroshot)

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

        if self.metadata['corpus_name'] == "EU_projects":

            ids_corpus = self.df_corpus['id']
            self.class_name = 'AIimported'
            # Import ids of docs from the positive class
            ids_pos, msg = self.DM.import_labels(
                ids_corpus=ids_corpus, tag=self.class_name)

            # Generate dataset dataframe
            self.df_dataset = self.CorpusProc.make_PU_dataset(ids_pos)
            self.DM.save_dataset(
                self.df_dataset, tag=self.class_name, save_csv=True)
        else:
            logging.error("-- No label files available for importation from "
                          f"corpus {self.metadata['corpus_name']}")
            msg = " "

        return msg

    def analyze_keywords(self, wt=2):
        """
        Get a set of positive labels using keyword-based search

        Parameters
        ----------
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

        y = self.CorpusProc.score_by_keywords(
            self.keywords, wt=wt, method="count")

        # Plot sorted document scores
        plotter.plot_doc_scores(y)

        return y, df_stats, kf_stats

    def get_labels_by_keywords(self, wt=2, n_max=2000, s_min=1, tag="kwds",
                               method='count'):
        """
        Get a set of positive labels using keyword-based search

        Parameters
        ----------
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
        method: 'embedding' or 'count', optional
            Selection method: 'count' (based on counting occurences of keywords
            in docs) or 'embedding' (based on the computation of similarities
            between doc and keyword embeddings)
        """

        logging.info(f'-- Selected keywords: {self.keywords}')

        # Take name of the SBERT model from the configuration parameters
        model_name = self.global_parameters['keywords']['model_name']

        # Find the documents with the highest scores given the keywords
        ids, scores = self.CorpusProc.filter_by_keywords(
            self.keywords, wt=wt, n_max=n_max, s_min=s_min,
            model_name=model_name, method=method)

        # Set the working class
        self.class_name = tag
        # Generate dataset dataframe
        self.df_dataset = self.CorpusProc.make_PU_dataset(ids, scores)

        # ############
        # Save dataset
        msg = self.DM.save_dataset(
            self.df_dataset, tag=self.class_name, save_csv=True)

        # ################################
        # Save parameters in metadata file
        key = 'keyword_based_label_parameters'
        if key not in self.metadata:
            self.metadata[key] = {}
        self.metadata[key][tag] = {
            'wt': wt,
            'n_max': n_max,
            's_min': s_min,
            'keywords': self.keywords}

        # Metadata for evaluation
        # FIXME: the code below is not used. To be moved to another method.
        eval_scores = False
        if eval_scores:
            # Save tpr fpr and ROC curve
            # FIXME: The name of the SBERT model should be read from the config
            # file (parameters.default.yaml or metadata file (metadata.yaml))
            model_name = 'all-MiniLM-L6-v2'
            results_out_fname = f'results_{model_name}_{self.keywords}_ROC'
            results_fname = self.path2labels / results_out_fname
            np.savez(results_fname, tpr_roc=eval_scores['tpr_roc'],
                     fpr_roc=eval_scores['fpr_roc'])

            fig, ax = plt.subplots()
            plt.plot(eval_scores['fpr_roc'], eval_scores['tpr_roc'],
                     lw=2.5, label=self.keywords)
            plt.grid(b=True, which='major', color='gray', alpha=0.6,
                     linestyle='dotted', lw=1.5)
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('ROC curve')
            plt.legend()
            figure_out_fname = f'figure_{model_name}_{self.keywords}_ROC'
            figure_fname = self.path2labels / figure_out_fname
            plt.savefig(figure_fname)

            # Save smin and nmax evaluation scores
            del eval_scores['fpr_roc'], eval_scores['tpr_roc']
            self.metadata[key][tag].__setitem__('eval_scores', eval_scores)

        self._save_metadata()

        return msg

    def get_labels_by_zeroshot(self, n_max=2000, s_min=0.1, tag="zeroshot"):
        """
        Get a set of positive labels using a zero-shot classification model

        Parameters
        ----------
        n_max: int or None, optional (defaul=2000)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=0.1)
            Minimum score. Only elements strictly above s_min are selected
        tag: str, optional (default=1)
            Name of the output label set.
        """

        # Filter documents by zero-shot classification
        ids, scores = self.CorpusProc.filter_by_zeroshot(
            self.keywords, n_max=n_max, s_min=s_min)

        # Set the working class
        self.class_name = tag

        # Generate dataset dataframe
        self.df_dataset = self.CorpusProc.make_PU_dataset(ids, scores)

        # ############
        # Save dataset
        msg = self.DM.save_dataset(
            self.df_dataset, tag=self.class_name, save_csv=True)

        # ################################
        # Save parameters in metadata file
        key = 'zeroshot_parameters'
        if key not in self.metadata:
            self.metadata[key] = {}
        self.metadata[key][tag] = {
            'keyword': self.keywords,
            'n_max': n_max,
            's_min': s_min}
        self._save_metadata()

        return msg

    def get_labels_by_topics(self, topic_weights, T, df_metadata, n_max=2000,
                             s_min=1, tag="tpcs"):
        """
        Get a set of positive labels from a weighted list of topics

        Parameters
        ----------
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
        ids, scores = self.CorpusProc.filter_by_topics(
            T, df_metadata['corpusid'], topic_weights, n_max=n_max,
            s_min=s_min)

        # Set the working class
        self.class_name = tag

        # Generate dataset dataframe
        self.df_dataset = self.CorpusProc.make_PU_dataset(ids, scores)

        # ############
        # Save dataset
        msg = self.DM.save_dataset(
            self.df_dataset, tag=self.class_name, save_csv=True)

        # ################################
        # Save parameters in metadata file
        key = 'topic_based_label_parameters'
        if key not in self.metadata:
            self.metadata[key] = {}
        self.metadata[key][tag] = {
            'topic_weights': topic_weights,
            'n_max': n_max,
            's_min': s_min}
        self._save_metadata()

        return msg

    def evaluate_PUlabels(self, gold_standard):
        """
        Evaluate the current set of PU labels
        """

        if self.df_dataset is None:
            logging.warning("-- No labels loaded. "
                            "You must load or create a set of labels first")
            return

        if gold_standard != NO_GOLD_STANDARD:
            pass

        p2fig = self.path2output / f'{self.class_name}_sorted_PUscores.png'
        y = self.df_dataset.base_scores
        plotter.plot_doc_scores(y, path2figure=p2fig)

        breakpoint()
        print("WORK IN PROGRESS")
        return

    def load_labels(self, class_name):
        """
        Load a set of labels and its corresponding dataset (if it exists)

        Parameters
        ----------
        class_name : str
            Name of the target category
        """

        self.class_name = class_name

        # Load dataset
        self.df_dataset, msg = self.DM.load_dataset(self.class_name)

        # If a model has been already trained for the given class, load it.
        if self._is_model(verbose=False):

            logging.info("-- Loading classification model")
            path2model = self.path2models / self.class_name
            model_type = self.global_parameters['classifier']['model_type']
            model_name = self.global_parameters['classifier']['model_name']
            breakpoint()
            self.dc = CorpusClassifier(
                self.df_dataset, model_type=model_type, model_name=model_name,
                path2transformers=path2model)
            self.dc.load_model()

        else:
            # No model trained for this class
            self.dc = None

        return msg

    def reset_labels(self, labelset):
        """
        Reset all labels and models associated to a given category

        Parameters
        ----------
        labelset: str
            Name of the category to be removed.
        """

        # Remove files
        self.DM.reset_labels(tag=labelset)

        # Remove label info from metadata, it it exist
        for key in ['keyword_based_label_parameters',
                    'topic_based_label_parameters',
                    'zero_shot_parameters']:
            if key in self.metadata and labelset in self.metadata[key]:
                self.metadata[key].pop(labelset, None)

        self._save_metadata()

        return

    def train_PUmodel(self, max_imbalance=3, nmax=400):
        """
        Train a domain classifiers

        Parameters
        ----------
        max_imbalance : int or float or None, optional (default=None)
            Maximum ratio negative vs positive samples. If the ratio in
            df_dataset is higher, the negative class is subsampled.
            If None, the original proportions are preserved
        nmax : int or None (defautl=None)
            Maximum size of the whole (train+test) dataset
        """

        if self.df_dataset is None:
            logging.warning("-- No labels loaded. "
                            "You must load or create a set of labels first")
            return

        # Labels from the PU dataset are stored in column "PUlabels". We must
        # copy them to column "labels" which is the name required by
        # simpletransformers
        self.df_dataset[['labels']] = self.df_dataset[['PUlabels']]

        path2model = self.path2models / self.class_name
        self.dc = CorpusClassifier(
            self.df_dataset,
            model_type=self.global_parameters['classifier']['model_type'],
            model_name=self.global_parameters['classifier']['model_name'],
            path2transformers=path2model)

        # Select data for training and testing
        self.dc.train_test_split(max_imbalance=max_imbalance, nmax=nmax,
                                 random_state=0)

        # Train the model using simpletransformers
        self.dc.train_model()

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

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Evaluate the model over the test set
        result, wrong_predictions = self.dc.eval_model(tag_score='PUscore')

        # Pretty print dictionary of results
        logging.info(f"-- Classification results: {result}")
        for r, v in result.items():
            logging.info(f"-- -- {r}: {v}")

        # Update dataset file to include scores
        self._save_dataset()

        return result

    def get_feedback(self):
        """
        Gets some labels from a user for a selected subset of documents
        """

        # Check if a classifier object exists
        if not self._is_model():
            return

        # STEP 1: Select bunch of documents at random
        selected_docs = self.dc.AL_sample(
            n_samples=self.global_parameters['active_learning']['n_docs'],
            sampler=self.global_parameters['active_learning']['sampler'],
            p_ratio=self.global_parameters['active_learning']['p_ratio'],
            top_prob=self.global_parameters['active_learning']['top_prob'])

        if selected_docs is None:
            return

        # Indices of the selected docs
        idx = selected_docs.index

        # STEP 2: Request labels
        labels = self.get_labels_from_docs(selected_docs)

        # STEP 3: Annotate
        self.dc.annotate(idx, labels)

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

    def retrain_model(self):
        """
        Improves classifier performance using the labels provided by users
        """

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Retrain model using the new labels
        self.dc.retrain_model()

        # Update status.
        # Since training takes much time, we store the classification results
        # in files
        self._save_dataset()
        self.state['trained_model'] = True
        self._save_metadata()

        return

    def reevaluate_model(self):
        """
        Evaluate a domain classifier
        """

        # FIXME: this code is equal to evaluate_model() but using a different
        #        tagscore. It should be modified to provide evaluation metrics
        #        computed from the annotated labels.

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Evaluate the model over the test set
        result, wrong_predictions = self.dc.eval_model(tag_score='PNscore')

        # Pretty print dictionary of results
        logging.info(f"-- Classification results: {result}")
        for r, v in result.items():
            logging.info(f"-- -- {r}: {v}")

        # Update dataset file to include scores
        self._save_dataset()

        return result


class TaskManagerCMD(TaskManager):
    """
    Provides extra functionality to the task manager, requesting parameters
    from users from a command window.
    """

    def __init__(self, path2project, path2source=None, path2zeroshot=None,
                 config_fname='parameters.yaml',
                 metadata_fname='metadata.yaml', set_logs=True):
        """
        Opens a task manager object.

        Parameters
        ----------
        path2project : pathlib.Path
            Path to the application project
        path2source : str or pathlib.Path or None (default=None)
            Path to the folder containing the data sources
        path2zeroshot : str or pathlib.Path or None (default=None)
            Path to the folder containing the zero-shot-model
        config_fname : str, optional (default='parameters.yaml')
            Name of the configuration file
        metadata_fname : str or None, optional (default=None)
            Name of the project metadata file.
            If None, no metadata file is used.
        set_logs : bool, optional (default=True)
            If True logger objects are created according to the parameters
            specified in the configuration file
        """

        super().__init__(
            path2project, path2source, path2zeroshot=path2zeroshot,
            config_fname=config_fname, metadata_fname=metadata_fname,
            set_logs=set_logs)

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
        kw_library = self.DM.get_keywords_list()
        # Ask keywords through the query manager
        keywords = self.QM.ask_keywords(kw_library)

        if keywords == ['__all_AI']:
            keywords = self.DM.get_keywords_list()

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

        Parameters
        ----------
        topic_words : list of str
            Description of each available topic as a list of its most relevant
            words

        Returns
        -------
        weighted_topics : list of tuple
            A weighted list of topics.
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

        # Get method
        method = self.QM.ask_value(
            query=("Set method: (e)mbedding (slow) or (c)ount (fast)"),
            convert_to=str,
            default=self.global_parameters['keywords']['method'])

        if method == 'e':
            method = 'embedding'
        elif method == 'c':
            method = 'count'

        # Get keywords and labels
        self.keywords = self._ask_keywords()
        tag = self._ask_label_tag()

        # ##########
        # Get labels
        super().get_labels_by_keywords(
            wt=wt, n_max=n_max, s_min=s_min, tag=tag, method=method)

        return

    def get_labels_by_zeroshot(self):
        """
        Get a set of positive labels using keyword-based search
        """

        # ##############
        # Get parameters

        # Get weight parameter (weight of title word wrt description words)
        n_max = self.QM.ask_value(
            query=("Set maximum number of returned documents"),
            convert_to=int,
            default=self.global_parameters['zeroshot']['n_max'])

        # Get score threshold
        s_min = self.QM.ask_value(
            query=("Set score_threshold"),
            convert_to=float,
            default=self.global_parameters['zeroshot']['s_min'])

        # Get keywords and labels
        self.keywords = self.QM.ask_keywords()
        # Transform list in a comma-separated string of keywords, which is
        # the format used by the zero-shot classifier
        self.keywords = ', '.join(self.keywords)
        tag = self._ask_label_tag()

        # ##########
        # Get labels
        msg = super().get_labels_by_zeroshot(
            n_max=n_max, s_min=s_min, tag=tag)

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

        if T is None:
            msg = "-- No topic model available for this corpus"
            return msg

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

        labels = []
        width = 80

        print(width * "=")
        print("-- SAMPLED DOCUMENTS FOR LABELING:")

        print(width * "=")
        for i, doc in selected_docs.iterrows():
            print(f"ID: {doc.id}")
            if self.metadata['corpus_name'] == 'EU_projects':
                # Locate document in corpus
                doc_corpus = self.df_corpus[self.df_corpus['id'] == doc.id]
                # Get and print title
                title = doc_corpus.iloc[0].title
                print(f"TITLE: {title}")
                # Get and print description
                descr = doc_corpus.iloc[0].description
                print(f"DESCRIPTION: {descr}")
            else:
                # Get and print text
                text = doc.text
                print(f"TEXT: {text}")
            # Get and print prediction
            if 'prediction' in doc:
                print(f"PREDICTED CLASS: {doc.prediction}")

            labels.append(self.QM.ask_label())
            print(width * "=")

        # Label confirmation: this is to confirm that the labeler did not make
        # (conciously) a mistake.
        if not self.QM.confirm():
            logging.info("-- Canceling: new labels removed.")
            labels = []

        return labels

    def train_PUmodel(self):
        """
        Train a domain classifier
        """

        # Get weight parameter (weight of title word wrt description words)
        max_imbalance = self.QM.ask_value(
            query=("Introduce the maximum ratio negative vs positive samples "
                   "in the training set"),
            convert_to=float,
            default=self.global_parameters['classifier']['max_imbalance'])

        # Get score threshold
        nmax = self.QM.ask_value(
            query=("Maximum number of documents in the training set."),
            convert_to=int,
            default=self.global_parameters['classifier']['nmax'])

        super().train_PUmodel(max_imbalance, nmax)

        return


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
        kw_library = self.DM.get_keywords_list()
        suggested_keywords = ', '.join(kw_library)
        logging.info(
            f"-- Suggested list of keywords: {', '.join(kw_library)}\n")

        return suggested_keywords

    def get_labels_by_keywords(self, keywords, wt, n_max, s_min, tag, method):
        """
        Get a set of positive labels using keyword-based search through the
        MainWindow

        Parameters
        ----------
        keywords : list of str
            List of keywords
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
        n_max : int or None, optional (default=2000)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no limit
        s_min : float, optional (default=1)
            Minimum score. Only elements strictly above s_min are selected
        tag : str, optional (default=1)
            Name of the output label set.
        method : 'embedding' or 'count', optional
            Selection method: 'count' (based on counting occurrences of
            keywords in docs) or 'embedding' (based on the computation of
            similarities between doc and keyword embeddings)
        """

        # Keywords are received as arguments
        self.keywords = keywords

        # ##########
        # Get labels
        msg = super().get_labels_by_keywords(
            wt=wt, n_max=n_max, s_min=s_min, tag=tag, method=method)

        return msg

    def get_topic_words(self):
        """
        Get a set of positive labels from a weighted list of topics
        """

        # Load topics
        T, df_metadata, topic_words = self.DM.load_topics()

        # Remove all documents (rows) from the topic matrix, that are not
        # in self.df_corpus.
        if T is not None:
            T, df_metadata = self.CorpusProc.remove_docs_from_topics(
                T, df_metadata, col_id='corpusid')

        return topic_words, T, df_metadata

    def get_feedback(self, idx, labels):
        """
        Gets some labels from a user for a selected subset of documents

        Notes
        -----
        In comparison to the corresponding parent method, STEPS 1 and 2 are
        carried out directly through the GUI
        """

        # STEP 3: Annotate
        self.dc.annotate(idx, labels)

        # Update dataset file to include new labels
        self._save_dataset()

        return

    def train_PUmodel(self, max_imabalance, nmax):
        """
        Train a domain classifier

        Parameters
        ----------
        max_imabalance : int (default 3)
            Maximum ratio negative vs positive samples in the training set
        nmax : int (default = 400)
            Maximum number of documents in the training set.
        """

        super().train_PUmodel(max_imabalance, nmax)

        return

    def get_labels_by_zeroshot(self, keywords, n_max, s_min, tag):
        """
        Get a set of positive labels using a zero-shot classification model

        Parameters
        ----------
        keywords : list of str
            List of keywords
        n_max : int or None, optional (defaul=2000)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min : float, optional (default=0.1)
            Minimum score. Only elements strictly above s_min are selected
        tag : str, optional (default=1)
            Name of the output label set.
        """

        # Keywords, parameters and tag  are received as arguments
        self.keywords = keywords

        # Get labels
        msg = super().get_labels_by_zeroshot(
            n_max=n_max, s_min=s_min, tag=tag)

        logging.info(msg)

        return msg
