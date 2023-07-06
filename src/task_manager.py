"""
Defines classes that define methods to run the main tasks in the project,
using the core processing classes and methods.

@author: J. Cid-Sueiro, L. Calvo-Bartolome, A. Gallardo-Antolin, T.Ahlers
"""

import logging
import numpy as np
import pandas as pd

# Local imports
# You might need to update the location of the baseTaskManager class
from .base_taskmanager import baseTaskManager
from .data_manager import LogicalDataManager
from .data_manager import LocalDataManager
from .query_manager import QueryManager
from .domain_classifier.preprocessor import CorpusDFProcessor
from .domain_classifier.classifier import CorpusClassifier
from .domain_classifier.classifier import CorpusClassifierMLP
from .utils import plotter

# A message that is used twice in different parts of the code. It is defined
# here because the same message must be used in both cases.
NO_GOLD_STANDARD = 'Do not use a Gold Standard.'

# Name of the column of annotations in the datasets of manual labels
ANNOTATIONS = 'annotations'


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
                 metadata_fname='metadata.yaml', set_logs=True,
                 logical_dm=False):
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
        logical_dm : bool, optional (default=True)
            It True, a logical data manager is used
            If False, a local data manager is used
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
        self.corpus_has_embeddings = False

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
                         'corpus': 'corpus'        # Used by the IMT only
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

        # Path to the folder contaxining the zero-shot model
        self.path2zeroshot = path2zeroshot

        # Corpus dataframe
        self.df_corpus = None      # Corpus dataframe
        self.class_name = None     # Name of the working category
        self.keywords = None
        self.CorpusProc = None

        # CorpusClassifier object.
        self.dc = None

        # Temporary datasets of inputs, labels and outputs
        # WARNING: the content of this attribute will be a dataframe that
        #          will be loaded into the corpus_classifier object (self.dc).
        #          There, the dataset (in self.dc.dataset) will be modified
        #          and enriched with labels and predictions. The most updated
        #          version of the dataset should be taken from self.dc.dataset,
        #          when available.
        self.df_dataset = None

        # Extend base variables (defined in the base class) for state and
        # metadata with additional fields
        self.state['selected_corpus'] = False  # True if corpus was selected
        self.metadata['corpus_name'] = None

        # Datamanager
        if logical_dm:
            self.DM = LogicalDataManager(
                self.path2source, self.path2datasets, self.path2models,
                self.path2project, self, self.path2embeddings)
        else:
            self.DM = LocalDataManager(
                self.path2source, self.path2datasets, self.path2models,
                self.path2embeddings)

        return

    def _is_corpus(self, verbose=True):
        """
        Check if a corpus has been loaded.
        """

        # Just to abbreviate
        is_corpus = self.metadata['corpus_name'] is not None
        if not is_corpus:
            logging.warning("\n")
            logging.warning(
                "-- No corpus loaded. You must load a corpus first")

        return is_corpus

    def _is_model(self, verbose=True):
        """
        Check if labels have been loaded and a domain classifier object has
        been created.
        """

        if self.df_dataset is None:
            if verbose:
                logging.warning("-- No model is loaded. You must load or "
                                "create a set of labels first")
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
        Returns the list of available datasets
        """

        if self._is_corpus():
            dataset_list = self.DM.get_dataset_list()
        else:
            dataset_list = []

        return dataset_list

    # def _get_inference(self):
    #     """
    #     Returns inference manager options
    #     """
    #     corpus_has_embeddings = self.corpus_has_embeddings
    #     # corpus_has_embeddings = (
    #           self.DM.get_metadata()['corpus_has_embeddings']
    #     return ['Inference MLP'] if corpus_has_embeddings else []

    def inference(self, option=[]):
        """
        Infers data

        Parameters
        ----------
        option:
            Unused
        """

        if self.dc is None or self.dc.df_dataset is None:
            logging.warning("-- No model is loaded. "
                            "You must load or create a set of labels first")
            return

        metadata = self.DM.get_metadata()

        if 'corpus' in metadata:
            dPaths = {'d_documentEmbeddings': metadata['corpus'],
                      'p_prediction': self.path2output / self.class_name}
            self.dc.inferData(dPaths)
        else:
            logging.warning("-- Option not available for this corpus")

        return

    def _get_annotation_list(self):
        """
        Returns the list of available files with class annotations
        (i.e. class labels, likely obtained from annotations by humans in
        previous active learning sessions)
        """

        # Just to abbreviate
        corpus_name = self.metadata['corpus_name']

        if corpus_name is None:
            logging.warning("\n")
            logging.warning(
                "-- No corpus loaded. You must load a corpus first")
            annotations_list = []
        else:
            annotations_list = self.DM.get_annotation_list()

        return annotations_list

    def _get_gold_standard_labels(self):
        """
        Returns the list of gold-standard labels or labelsets available
        in the current corpus.

        Gold-standard labels are those whose name starts with 'target_'

        Gold-standard labels will be used for evaluation only, not for
        learning.
        """

        if not self._is_corpus():
            return []

        gs_labels = [x for x in self.df_corpus.columns
                     if x.startswith('target_')]

        if self.dc is not None and ANNOTATIONS in self.dc.df_dataset:
            gs_labels.append(ANNOTATIONS)

        if gs_labels == []:
            logging.warning('No Gold Standard is available. Please choose the '
                            'unique option in the menu')
            gs_labels = [NO_GOLD_STANDARD]

        return gs_labels

    def _convert_keywords(self, keywords, out='list'):
        """
        Converts a string variable keywords to the required format.

        If keywords is an empty string, self.keywords is returned.

        Otherwise, extra blank spaces are removed and, if needed, a comma
        separated string is transformed into a comma-separated list

        Parameters
        ----------
        keywords : str
            If "", self.keywords is returned
            Otherwise, a comma separated string of keywords is expected.
        out : str in {'list', 'str'}
            If list, a comma separated list of keywords is returned
            If str, a comma separated list of keywords is returned
        """

        if keywords == "":
            return self.keywords
        else:
            # Transform string into list of keywords
            keywords = keywords.split(",")
            # Clean extra blank spaces keyword-by-keyword
            keywords = [" ".join(k.split()) for k in keywords]

            # Join keywords back into a string
            if out == 'str':
                keywords = ", ".join(keywords)

            return keywords

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

        logging.info("-- Configuration file activated")

        return

    def load_corpus(self, corpus_name: str):
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
        sf = 1
        if corpus_name in sampling_factors:
            sf = sampling_factors[corpus_name]

        # The corpus cannot be changed inside the same project. If a corpus
        # was used before we must keep the same one.
        current_corpus = self.metadata['corpus_name']
        if (self.state['selected_corpus'] and corpus_name != current_corpus):
            logging.error(
                f"-- The corpus of this project is {current_corpus}. "
                f"Run another project to use {corpus_name}")
            return

        # Load corpus in a dataframe.
        self.df_corpus = self.DM.load_corpus(corpus_name, sampling_factor=sf)

        self.corpus_has_embeddings = 'embeddings' in self.df_corpus.columns

        self.CorpusProc = CorpusDFProcessor(
            self.df_corpus, self.path2embeddings, self.path2zeroshot)

        if not self.state['selected_corpus']:
            # Store the name of the corpus an object attribute because later
            # tasks will be referred to this corpus
            self.metadata['corpus_name'] = str(corpus_name)
            self.state['selected_corpus'] = True
            self._save_metadata()

        return

    def import_AI_subcorpus(self):
        """
        Import a subcorpus of documents related to AI.

        This method is very specific. Loads a subcorpus from EU_projects that
        is available from file. Not to be used for other corpora or other
        target domains.
        """

        if self.metadata['corpus_name'] == "EU_projects":

            ids_corpus = self.df_corpus['id']
            self.class_name = 'AIimported'
            # Import ids of docs from the positive class
            ids_pos, msg = self.DM.import_AI_subcorpus(
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

    def analyze_keywords(self, wt: float = 2.0, keywords: str = ""):
        """
        Get a set of positive labels using keyword-based search

        Parameters
        ----------
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
        keywords : str, optional (default= "")
            A comma-separated string of keywords.
            If the string is empty, the keywords are read from self.keywords
        """

        # Read keywords:
        self.keywords = self._convert_keywords(keywords, out='list')
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

    def get_labels_by_keywords(self, wt: float = 2.0, n_max: int = 50_000,
                               s_min: float = 1.0, tag: str = "kwds",
                               method: str = 'count', keywords: str = ""):
        """
        Get a set of positive labels using keyword-based search

        Parameters
        ----------
        wt : float, optional (default=2)
            Weighting factor for the title components. Keywords in the title
            are weighted by this factor
        n_max: int or None, optional (default=10_000)
            Maximum number of elements in the output list.
        s_min: float, optional (default=1)
            Minimum score. Only elements strictly above s_min are selected
        tag: str, optional (default='kwds')
            Name of the output label set.
        method: 'embedding' or 'count', optional
            Selection method: 'count' (based on counting occurences of keywords
            in docs) or 'embedding' (based on the computation of similarities
            between doc and keyword embeddings)
        keywords : str, optional (default="")
            A comma-separated string of keywords.
            If the string is empty, the keywords are read from self.keywords
        """

        # Check if corpus has been loaded
        if not self._is_corpus():
            return "No corpus has been loaded"

        # Read keywords:
        self.keywords = self._convert_keywords(keywords, out='list')
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
        # (note that we do not call self._save_dataset() here, because we
        #  are saving self.df_dataset, and not self.dc.df_dataset (the
        #  classifier object has not been created yet))
        msg = self.DM.save_dataset(
            self.df_dataset, tag=self.class_name, save_csv=True)

        # ################################
        # Save parameters in metadata file
        self.metadata[tag] = {
            'doc_selection': {
                'method': method,
                'wt': wt,
                'n_max': n_max,
                's_min': s_min,
                'keywords': self.keywords}}
        if method == 'embedding':
            self.metadata[tag]['doc_selection']['model'] = model_name

        self._save_metadata()

        return msg

    def get_labels_by_zeroshot(self, n_max: int = 2000, s_min: float = 0.1,
                               tag: str = "zeroshot", keywords: str = ""):
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
        keywords : str, optional (default="")
            A comma-separated string of keywords.
            If the string is empty, the keywords are read from self.keywords
        """

        # Check if corpus has been loaded
        if not self._is_corpus():
            return "No corpus has been loaded"

        # Read keywords:
        self.keywords = self._convert_keywords(keywords, out='list')
        logging.info(f'-- Selected keyword: {self.keywords}')

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
        self.metadata[tag] = {
            'doc_selection': {
                'method': 'zeroshot',
                'keyword': self.keywords,
                'n_max': n_max,
                's_min': s_min}}
        self._save_metadata()

        return msg

    def get_labels_by_topics(self, topic_weights, n_max: int = 2000,
                             s_min: float = 1.0, tag: str = "tpcs"):
        """
        Get a set of positive labels from a weighted list of topics

        Parameters
        ----------
        topic_weights: str or dict
            If dict, a dictionwary topics: weighs
            If str, a string of comma-separated topicis and wieighs:
            t1, w1, t2, w2, ...
        n_max: int or None, optional (defaul=2000)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=1)
            Minimum score. Only elements strictly above s_min are selected
        tag: str, optional (default=1)
            Name of the output label set.
        """

        # Check if corpus has been loaded
        if not self._is_corpus():
            return "No corpus has been loaded"

        # It topic_weights is a string, convert to dictionary
        if isinstance(topic_weights, str):
            topic_weights = QueryManager.str2dict(topic_weights)

        # Load topics
        T, df_metadata, topic_words = self.DM.load_topics()

        if T is None:
            msg = "-- No topic model available for this corpus"
            return msg

        # Remove all documents (rows) from the topic matrix, that are not
        # in self.df_corpus.
        T, df_metadata = self.CorpusProc.remove_docs_from_topics(
            T, df_metadata, col_id='corpusid')

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
        self.metadata[tag] = {
            'doc_selection': {
                'method': 'filter by topics',
                'topic_weights': topic_weights,
                'n_max': n_max,
                's_min': s_min}}

        self._save_metadata()

        return msg

    def evaluate_PUlabels(self, true_label_name: str):
        """
        Evaluate the current set of PU labels
        """

        if self.dc is None or self.dc.df_dataset is None:
            logging.warning("-- No model is loaded. "
                            "You must load or create a set of labels first")
            return

        if true_label_name != NO_GOLD_STANDARD:
            if true_label_name not in self.dc.df_dataset.columns:
                logging.warning("-- Gold standard not available in the "
                                "dataframe")
            else:
                # Test PU labels against annotations
                logging.info(
                    f"-- Quality of the PU labels wrt {true_label_name}")
                # self._label2label_metrics("PUlabels", true_label_name, "all")

                # Compute train and test metrics
                bmetrics, roc = self.dc.performance_metrics(
                    'PUlabels', true_label_name, 'all', pred_name="PUlabels",
                    score_name="base_scores", use_sampling_probs=True)

                # Plot train and test ROCs
                fname = (f'{self.class_name}_PUlabels_vs_{true_label_name}'
                         f'_ROC_all.png')
                p2fig = self.path2output / fname
                plotter.plot_roc(roc, bmetrics, tag='all', path2figure=p2fig)

                # Add AUC in roc to the metrics dictionary:
                if bmetrics is not None and roc is not None:
                    bmetrics['AUC'] = roc['auc']
                    if 'unweighted' in bmetrics and 'unweighted' in roc:
                        bmetrics['unweighted']['AUC'] = (
                            roc['unweighted']['auc'])

                # Save metrics in metadata file.
                if 'metrics' not in self.metadata[self.class_name]:
                    self.metadata[self.class_name]['metrics'] = {}
                key = f'PUlabels_vs_{true_label_name}'
                if key not in self.metadata[self.class_name]['metrics']:
                    self.metadata[self.class_name]['metrics'][key] = {}
                self.metadata[self.class_name]['metrics'][key]['all'] = (
                    bmetrics)

                self._save_metadata()

        # Plot score distributions (this does not depend on the gold standard)
        p2fig = self.path2output / f'{self.class_name}_PUscores.png'
        scores = self.df_dataset.base_scores
        n_pos = np.sum(self.df_dataset.PUlabels == 1)
        plotter.plot_doc_scores(scores, n_pos, path2figure=p2fig)

        return

    def load_labels(self, class_name, model_type=None, model_name=None):
        """
        Load a set of labels and its corresponding dataset (if it exists)

        Parameters
        ----------
        class_name : str
            Name of the target category
        model_type : str
            Type of classifier model
        model_name : str
            Name of the specific classifier model
        """

        self.class_name = class_name

        # Load dataset
        self.df_dataset, msg = self.DM.load_dataset(self.class_name)

        # If a model has been already trained for the given class, load it.
        if self._is_model(verbose=False):

            logging.info("-- Loading classification model")
            path2model = self.path2models / self.class_name
            if model_type is None:
                model_type = self.global_parameters['classifier']['model_type']
            if model_name is None:
                model_name = self.global_parameters['classifier']['model_name']

            # if self.DM.get_metadata()['corpus_has_embeddings']:
            if self.corpus_has_embeddings:

                self.df_dataset = (
                    self.CorpusProc.enrich_dataset_with_embeddings(
                        self.df_dataset, self.df_corpus))

                self.dc = CorpusClassifierMLP(
                    self.df_dataset, model_type=model_type,
                    model_name=model_name, path2transformers=path2model)
            else:

                self.dc = CorpusClassifier(
                    self.df_dataset, model_type=model_type,
                    model_name=model_name, path2transformers=path2model)

            self.dc.load_model()

        else:
            # No model trained for this class
            self.dc = None

        return msg

    def reset_labels(self, class_name):
        """
        Reset all labels and models associated to a given category

        Parameters
        ----------
        labelset: str
            Name of the category to be removed.
        """

        # Remove files
        self.DM.reset_labels(tag=class_name)

        # Remove label info from metadata, if it exist
        if class_name in self.metadata:
            self.metadata.pop(class_name, None)

        self._save_metadata()

        return

    def train_PUmodel(self, max_imbalance: float = 3.0, nmax: int = 400,
                      epochs: int = 3, freeze_encoder: bool = None,
                      batch_size: int = None, model_type: str = None,
                      model_name: str = None):
        """
        Train a domain classifier

        Parameters
        ----------
        max_imbalance : float, optional (default=3.0)
            Maximum ratio negative vs positive samples. If the ratio in
            df_dataset is higher, the negative class is subsampled.
            If None, the original proportions are preserved
        nmax : int, optional (defautl=400)
            Maximum size of the whole (train+test) dataset
        epochs : int, optional (default=3)
            Number of training epoch
        """

        if self.df_dataset is None:
            logging.warning("-- No model is loaded. "
                            "You must load or create a set of labels first")
            return

        # Configuration parameters
        params = self.global_parameters['classifier']  # Just to abbreviate
        if freeze_encoder is None:
            freeze_encoder = params['freeze_encoder']
        if batch_size is None:
            batch_size = params['batch_size']
        if model_type is None:
            model_type = params['model_type']
        if model_name is None:
            model_name = params['model_name']

        if self.dc is not None:
            # If there exists a classifier object, update the local dataset,
            # to generate the new classifier objetc.
            # This is to prevent label losses if the user trains a model,
            # annotates labels and then trains for a second time.
            self.df_dataset = self.dc.df_dataset

        # Labels from the PU dataset are stored in column "PUlabels". We must
        # copy them to column "labels" which is the name required by
        # simpletransformers
        self.df_dataset[['labels']] = self.df_dataset[['PUlabels']]

        path2model = self.path2models / self.class_name

        # if self.DM.get_metadata()['corpus_has_embeddings']:
        if self.corpus_has_embeddings:

            self.df_dataset = self.CorpusProc.enrich_dataset_with_embeddings(
                self.df_dataset, self.df_corpus)

            self.dc = CorpusClassifierMLP(
                self.df_dataset, model_type=model_type, model_name=model_name,
                path2transformers=path2model)
        else:
            self.dc = CorpusClassifier(
                self.df_dataset, model_type=model_type, model_name=model_name,
                path2transformers=path2model)

        # Select data for training and testing
        self.dc.train_test_split(max_imbalance=max_imbalance, nmax=nmax,
                                 random_state=0)

        # ['id', 'text', 'base_scores', 'PUlabels', 'labels', 'train_test']
        # Train the model using simpletransformers
        train_ok = self.dc.train_model(
            epochs=epochs, validate=True, freeze_encoder=freeze_encoder,
            tag="PU", batch_size=batch_size)

        if train_ok:
            # Update status.
            # Since training takes much time, we store the classification
            # results in files
            self._save_dataset()
            self.state['trained_model'] = True

            self.metadata[self.class_name]['PU_training'] = {
                'model_type': model_type,
                'model_name': model_name,
                'freeze_encoder': freeze_encoder,
                'max_imbalance': max_imbalance,
                'nmax': nmax,
                'epochs': epochs,
                'best_epoch': self.dc.best_epoch}
            self._save_metadata()

        else:
            logging.warning(
                "-- The model could not be trained. Maybe you can make a "
                "feasible dataset by get more documents from the positive "
                "class")
        return

    def evaluate_PUmodel(self, samples: str = "train_test"):
        """
        Evaluate a domain classifiers
        """

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Configuration parameters
        batch_size = self.global_parameters['classifier']['batch_size']

        # Evaluate the model over the test set
        result, wrong_predictions = self.dc.eval_model(
            samples=samples, tag='PU', batch_size=batch_size)

        # Pretty print dictionary of results
        logging.info(f"-- Classification results: {result}")
        for r, v in result.items():
            logging.info(f"-- -- {r}: {v}")

        # Update dataset file to include scores
        self._save_dataset()

        return result

    def _performance_metrics(self, tag_model, true_label_name, subset,
                             use_sampling_probs=True):
        """
        Compute all performance metrics based on the data available at the
        current dataset.

        Parameters
        ----------
        tag_model : str in {'PU', 'PN'}
            An ettiquete that identifies the model to be evaluated.
        true_label_name : str
            Name of the column in the dataset containing the "true labels" to
            be used as a reference for evaluation
        """

        # Compute train and test metrics
        bmetrics, roc = self.dc.performance_metrics(
            tag_model, true_label_name, subset,
            use_sampling_probs=use_sampling_probs)

        # Plot train and test ROCs
        fname = (f'{self.class_name}_{tag_model}_vs_{true_label_name}_ROC_'
                 f'{subset}.png')
        p2fig = self.path2output / fname
        plotter.plot_roc(roc, bmetrics, tag=subset, path2figure=p2fig)

        # Add AUC in roc to the metrics dictionary:
        if bmetrics is not None and roc is not None:
            bmetrics['AUC'] = roc['auc']
            if 'unweighted' in bmetrics and 'unweighted' in roc:
                bmetrics['unweighted']['AUC'] = roc['unweighted']['auc']

        # Save metrics in metadata file.
        if 'metrics' not in self.metadata[self.class_name]:
            self.metadata[self.class_name]['metrics'] = {}
        key = f'{tag_model}_vs_{true_label_name}'
        if key not in self.metadata[self.class_name]['metrics']:
            self.metadata[self.class_name]['metrics'][key] = {}
        self.metadata[self.class_name]['metrics'][key][subset] = bmetrics

        self._save_metadata()

        return

    def _label2label_metrics(self, tag_model, true_label_name, subset,
                             use_sampling_probs=True):
        """
        Compute all performance metrics based on the data available at the
        current dataset.

        Parameters
        ----------
        tag_model : str in {'PU', 'PN'}
            An ettiquete that identifies the model to be evaluated.
        true_label_name : str
            Name of the column in the dataset containing the "true labels" to
            be used as a reference for evaluation
        """

        # Compute train and test metrics
        metrics = self.dc.label2label_metrics(
            tag_model, true_label_name, subset,
            use_sampling_probs=use_sampling_probs)

        # Save metrics in metadata file.
        if 'metrics' not in self.metadata[self.class_name]:
            self.metadata[self.class_name]['metrics'] = {}
        key = f'{tag_model}_vs_{true_label_name}'
        if key not in self.metadata[self.class_name]:
            self.metadata[self.class_name]['metrics'][key] = {}
        self.metadata[self.class_name]['metrics'][key][subset] = metrics

        self._save_metadata()

        return

    def performance_metrics_PU(self):
        """
        Compute all performance metrics for the PU model, based on the data
        available at the current dataset
        """

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Test PU predictions against PUlabels
        logging.info("-- Quality of the PU predictor wrt the PU labels")
        self._performance_metrics("PU", "PUlabels", "train")
        self._performance_metrics("PU", "PUlabels", "test")
        self._performance_metrics("PU", "PUlabels", "all",
                                  use_sampling_probs=False)

        if ANNOTATIONS in self.dc.df_dataset:
            # Test PU predictions against annotations
            self._performance_metrics("PU", ANNOTATIONS, "test")
            self._performance_metrics("PU", ANNOTATIONS, "unused")
            self._performance_metrics("PU", ANNOTATIONS, "all")

        return

    def performance_metrics_PN(self):
        """
        Compute all performance metrics based on the data available at the
        current dataset.
        """

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Test PN predictions against PUlabels
        self._performance_metrics("PN", "PUlabels", "train")
        self._performance_metrics("PN", "PUlabels", "test")
        self._performance_metrics("PN", "PUlabels", "all",
                                  use_sampling_probs=False)

        # Test PN predictions against annotations
        self._performance_metrics("PN", ANNOTATIONS, "test")
        self._performance_metrics("PN", ANNOTATIONS, "unused")
        self._performance_metrics("PN", ANNOTATIONS, "all")

        return

    def get_feedback(self, sampler: str = ""):
        """
        Gets some labels from a user for a selected subset of documents

        Parameters
        ----------
        sampler : str, optional (default = "")
            Type of sampler. If "", the sampler is read from the global
            parameters
        """

        # This is for compatibility with the GUI
        if sampler == "" or sampler is None:
            sampler = self.global_parameters['active_learning']['sampler']

        # Check if a classifier object exists
        if not self._is_model():
            return

        # STEP 1: Select bunch of documents at random
        selected_docs = self.dc.AL_sample(
            n_samples=self.global_parameters['active_learning']['n_docs'],
            sampler=sampler,
            p_ratio=self.global_parameters['active_learning']['p_ratio'],
            top_prob=self.global_parameters['active_learning']['top_prob'])

        if selected_docs is None:
            return

        # Indices of the selected docs
        idx = selected_docs.index

        # TODO: Save selected_docs
        self.DM.save_selected_docs(selected_docs, tag=self.class_name)

        # STEP 2: Request labels
        labels = self.get_labels_from_docs()

        # STEP 3: Annotate
        self.dc.annotate(idx, labels, col=ANNOTATIONS)

        # Update dataset file to include new labels
        self._save_dataset()

        n_labels, n_train, n_test, n_unused = self.dc.num_annotations()
        logging.info("-- Summary of current annotations:")
        logging.info(f"-- -- Annotations: {n_labels}")
        logging.info(f"-- -- Train: {n_train}")
        logging.info(f"-- -- Test: {n_test}")
        logging.info(f"-- -- Unused: {n_unused}")

        return

    def sample_documents(self, sampler: str = "", fmt: str = "csv",
                         n_samples: int = -1):
        """
        Gets some labels from a user for a selected subset of documents

        Parameters
        ----------
        sampler : str, optional (default = "")
            Type of sampler. If "", the sampler is read from the global
            parameters
        fmt : str in {'csv', 'json'}, optional, default = "csv"
            Output file format
        n_samples : int, optional (default=-1)
            Number of samples to return.
            If -1, the number of samples is taken from the configuration file
        """

        # Set the number of documents to sample
        if n_samples == -1:
            n_samples = self.global_parameters['active_learning']['n_docs']

        # This is for compatibility with the GUI
        if sampler == "" or sampler is None:
            sampler = self.global_parameters['active_learning']['sampler']

        # Check if a classifier object exists
        if not self._is_model():
            return

        # STEP 1: Select bunch of documents at random
        selected_docs = self.dc.AL_sample(
            n_samples=n_samples, sampler=sampler,
            p_ratio=self.global_parameters['active_learning']['p_ratio'],
            top_prob=self.global_parameters['active_learning']['top_prob'])

        # TODO: Save selected_docs
        self.DM.save_selected_docs(selected_docs, tag=self.class_name, fmt=fmt)

        # Update dataset file (AL_sample changes data in some colums)
        self._save_dataset()

        return

    def get_labels_from_docs(self):
        """
        Requests feedback about the class of given documents.

        This method assumes user queryng through the command window. It should
        be overwrittend by inherited classes to adapt the specific UI of the
        application

        Returns
        -------
        labels : list of boolean
            Labels for the given documents, in the same order than the
            documents in the input dataframe
        """

        # Load sampled documents
        selected_docs = self.DM.load_selected_docs(tag=self.class_name)

        # Temporal query manager object
        QM = QueryManager()

        labels = []
        width = 80

        print(width * "=")
        print("-- SAMPLED DOCUMENTS FOR LABELING:")

        print(width * "=")
        k = 1    # A classic counter
        for i, doc in selected_docs.iterrows():
            print(f"Document {k} out of {len(selected_docs)}")
            k += 1
            print(f"ID: {doc.id}")
            # if self.metadata['corpus_name'] == 'EU_projects':
            if 'title' in self.df_corpus and 'description' in self.df_corpus:
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
            if 'prob_pred' in doc:
                print(f"SCORE: {doc.prob_pred}")

            labels.append(QM.ask_label())
            print(width * "=")
        # Label confirmation: this is to confirm that the labeler did not make
        # (consciously) a mistake.
        if not QM.confirm():
            logging.info("-- Canceling: new labels removed.")
            labels = []

        self.DM.save_new_labels(selected_docs.index, labels,
                                tag=self.class_name)
        logging.info("-- New labels saved.")

        return labels

    def annotate(self, fmt: str = "csv"):
        """
        Save user-provided labels in the dataset

        Parameters
        ----------
        fmt : str in {'csv', 'json'}, optional, default = "csv"
            Input file format
        """

        # Load sampled documents
        selected_docs = self.DM.load_selected_docs(
            tag=self.class_name, fmt=fmt)
        df_labels = self.DM.load_new_labels(tag=self.class_name, fmt=fmt)

        # Indices of the selected docs
        idx = selected_docs.index

        # Check consistency: the indices in selected_doc and df_labels must
        # be the same and in the same order. Otherwise, both dataframes could
        # correspond to different annotation rounds, and must be elliminated
        if set(df_labels.index) != set(idx):
            logging.error("-- The files of last sampled documents and last"
                          "labels do not match. Annotation aborted.")
            logging.error("-- You should re-sample and re-annotate")
            return
        labels = list(df_labels.labels)

        # STEP 3: Annotate
        self.dc.annotate(idx, labels, col=ANNOTATIONS)

        # Remove temporary label files
        self.DM.remove_temp_files(self.class_name)

        # Update dataset file to include new labels
        self._save_dataset()

        n_labels, n_train, n_test, n_unused = self.dc.num_annotations()
        logging.info("-- Summary of current annotations:")
        logging.info(f"-- -- Annotations: {n_labels}")
        logging.info(f"-- -- Train: {n_train}")
        logging.info(f"-- -- Test: {n_test}")
        logging.info(f"-- -- Unused: {n_unused}")

        return

    def retrain_model(self, epochs: int = 3):
        """
        Improves classifier performance using the labels provided by users
        """

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Configuration parameters
        batch_size = self.global_parameters['classifier']['batch_size']

        # Retrain model using the new labels
        self.dc.retrain_model(freeze_encoder=True, batch_size=batch_size,
                              epochs=epochs)

        # Update status.
        # Since training takes much time, we store the classification results
        # in files
        self._save_dataset()
        self.state['trained_model'] = True
        self._save_metadata()

        return

    def reevaluate_model(self, samples: str = "train_test"):
        """
        Evaluate a domain classifier
        """

        # FIXME: this code is equal to evaluate_model() but using a different
        #        tagscore. It should be modified to provide evaluation metrics
        #        computed from the annotated labels.

        # Check if a classifier object exists
        if not self._is_model():
            return

        # Configuration parameters
        batch_size = self.global_parameters['classifier']['batch_size']

        # Evaluate the model over the test set
        result, wrong_predictions = self.dc.eval_model(
            samples=samples, tag='PN', batch_size=batch_size)

        # Pretty print dictionary of results
        logging.info(f"-- Classification results: {result}")
        for r, v in result.items():
            logging.info(f"-- -- {r}: {v}")

        # Update dataset file to include scores
        self._save_dataset()

        return result

    def import_annotations(self, domain_name: str):
        """
        Imports / exports annotations from / to a file in the dataset folder.

        This will be useful to share annotations from different projects.

        Parameters
        ----------
        domain_name : str
            Name of the domain
        """

        # Check if a classifier object exists
        if self.dc is None:
            logging.error("-- No annotations to export. Load labels first")
            return

        df_annotations = self.DM.import_annotations(domain_name)

        # Integrate annotations into dataset...
        self.dc.update_annotations(df_annotations)
        self._save_dataset()

        return

    def export_annotations(self, domain_name: str):
        """
        Imports / exports annotations from / to a file in the dataset folder.

        This will be useful to share annotations from different projects.
        """

        if self.dc is None:
            logging.error("-- No annotations to export. Load labels first")
            return

        # Get the annotation sub-dataframe
        df_annotations = self.dc.get_annotations(annot_name=ANNOTATIONS)

        # Extract label dataframe from the dataset.
        logging.info("-- Saving annotations in source folder")
        self.DM.export_annotations(df_annotations, domain_name)

        return


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
            # Most AI keywords are read from a file, that misses a few
            # relevant keywords that are added here.
            keywords = (['artificial intelligence', 'argumentation framework',
                         'intelligent tutoring system',
                         'nonlinear archetypal analysis',
                         'non-linear archetypal analysis',
                         'random forest',
                         'rule based translation', 'rule-based translation',
                         'statistical machine translation', 'pytorch']
                        + self.DM.get_keywords_list())

            wrong_keywords = {
                'active learning',   # It's a more relevant topic in Ed. Sci.
                'statistical machinetranslation',  # Appears this way in metad.
                'rule based translation '}         # Extra space
            keywords = [k for k in keywords if k not in wrong_keywords]

            # This is to avoid keyword repetitions
            keywords = list(set(keywords))

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
        df_metadata, topic_words = self.DM.load_topic_metadata()

        # Ask for topic weights
        topic_weights = self._ask_topics(topic_words)
        # Ask tag for the label file
        tag = self._ask_label_tag()

        # ##########
        # Get labels
        msg = super().get_labels_by_topics(
            topic_weights, n_max=n_max, s_min=s_min, tag=tag)

        return msg

    def export_annotations(self):

        if self.dc is None:
            logging.error("-- No annotations to export. Load labels first")
            return

        # Get domain_name
        domain_name = self.QM.ask_value(
            query=(f"Write the domain name (for class {self.class_name}"),
            convert_to=str,
            default="unknown_domain")

        super().export_annotations(domain_name)

        return

    def train_PUmodel(self):
        """
        Train a domain classifier
        """

        max_imbalance = 1.0
        nmax = np.inf
        epochs = 0

        # if not self.DM.get_metadata()['corpus_has_embeddings']:
        if not self.corpus_has_embeddings:

            # Get weight parameter (weight of title word wrt description words)
            max_imbalance = self.QM.ask_value(
                query=("Introduce the maximum ratio negative vs positive "
                       "samples in the training set"),
                convert_to=float,
                default=self.global_parameters['classifier']['max_imbalance'])

            # Get score threshold
            nmax = self.QM.ask_value(
                query=("Maximum number of documents in the training set"),
                convert_to=int,
                default=self.global_parameters['classifier']['nmax'])

            # Get score threshold
            epochs = self.QM.ask_value(
                query=("Number of training epochs"),
                convert_to=int,
                default=self.global_parameters['classifier']['epochs'])

        super().train_PUmodel(max_imbalance, nmax, epochs)

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
        df_metadata, topic_words = self.DM.load_topic_metadata()

        return topic_words, df_metadata

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
        msg = super().get_labels_by_zeroshot(n_max=n_max, s_min=s_min, tag=tag)

        logging.info(msg)

        return msg


class TaskManagerIMT(TaskManager):
    """
    Provides extra functionality to the task manager, to be used by the
    Interactive Model Trainer of the IntelComp project.
    """

    def __init__(self, path2project, path2source=None, path2zeroshot=None):

        super().__init__(path2project, path2source, path2zeroshot,
                         logical_dm=True)

        self.project_folder = str(path2project).split('/')[-1]

    def get_labels_by_zeroshot(self, n_max: int = 2000, s_min: float = 0.1,
                               tag: str = "zeroshot", keywords: str = ""):
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
        keywords : str, optional (default="")
            A comma-separated string of keywords.
            If the string is empty, the keywords are read from self.keywords
        """

        # Check if corpus has been loaded
        if not self._is_corpus():
            return "No corpus has been loaded"

        msg = None
        try:
            df_dataset = self.DM.load_dataset(tag)[0]
            processed_ids = df_dataset['id'].to_numpy()
        except Exception:
            df_dataset = pd.DataFrame([])
            processed_ids = []
        while True:

            # Read keywords:
            self.keywords = self._convert_keywords(keywords, out='list')
            logging.info(f'-- Selected keyword: {self.keywords}')

            # Filter documents by zero-shot classification
            ids, scores = self.CorpusProc.filter_by_zeroshot(
                self.keywords, n_max=n_max, s_min=s_min,
                processed_ids=processed_ids)

            if len(scores) == 0:
                msg = '-- finished'
                break

            # Set the working class
            self.class_name = tag

            # Generate dataset dataframe
            self.df_dataset = self.CorpusProc.make_PU_dataset(ids, scores)

            # merge datasets
            self.df_dataset = pd.concat([df_dataset, self.df_dataset])
            self.df_dataset = self.df_dataset.reset_index(drop=True)

            # ############
            # Save dataset
            msg = self.DM.save_dataset(
                self.df_dataset, tag=self.class_name, save_csv=True)

            processed_ids = self.df_dataset['id'].to_numpy()
            df_dataset = self.df_dataset

            # ################################
            # Save parameters in metadata file
            self.metadata[tag] = {
                'doc_selection': {
                    'method': 'zeroshot',
                    'keyword': self.keywords,
                    'n_max': n_max,
                    's_min': s_min}}
            self._save_metadata()

        return msg

    def on_create_list_of_keywords(
            self, corpus_name: str, description: str = "",
            visibility: str = 'Private', wt: float = 2.0, n_max: int = 2000,
            s_min: float = 1.0, tag: str = "kwds", method: str = 'count',
            keyword_list: str = "", keywords: str = "",
            max_imbalance: float = 3.0, nmax: int = 400, epochs: int = 3,
            freeze_encoder: bool = True, batch_size: int = 8,
            model_type: str = 'mpnet',
            model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        on button click create with option: from list of keywords
        """

        self.setup()
        self.load_corpus(corpus_name)
        if keyword_list == '__all_AI':
            self.keywords = (
                ['artificial intelligence', 'argumentation framework',
                 'intelligent tutoring system',
                 'nonlinear archetypal analysis',
                 'non-linear archetypal analysis',
                 'random forest',
                 'rule based translation', 'rule-based translation',
                 'statistical machine translation', 'pytorch']
                + self.DM.get_keywords_list())
        if keyword_list == '':
            self.keywords = np.array(keywords.split(','))

        self.get_labels_by_keywords(wt, n_max, s_min, tag, method, keywords)
        self.train_PUmodel(
            max_imbalance, nmax, epochs, freeze_encoder=freeze_encoder,
            batch_size=batch_size, model_type=model_type,
            model_name=model_name)
        self.DM.save_model_json(
            self.project_folder, description, visibility, tag, 'Keyword-based',
            self.dc.config)

    def on_create_category_name(
            self, corpus_name: str, description: str = "",
            visibility: str = 'Private', n_max: int = 2000, s_min: float = 0.1,
            tag: str = "zeroshot", keywords: str = "",
            max_imbalance: float = 3.0, nmax: int = 400, epochs: int = 3,
            freeze_encoder: bool = True, batch_size: int = 8,
            model_type: str = 'mpnet',
            model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        on button click create with option: from topic selection function
        """

        self.setup()
        self.load_corpus(corpus_name)
        self.get_labels_by_zeroshot(n_max, s_min, tag, keywords)
        self.train_PUmodel(
            max_imbalance, nmax, epochs, freeze_encoder=freeze_encoder,
            batch_size=batch_size, model_type=model_type,
            model_name=model_name)
        self.DM.save_model_json(
            self.project_folder, description, visibility, tag, 'Keyword-based',
            self.dc.config)

    def on_create_topic_selection(
            self, corpus_name: str, description: str = "", tag: str = "topics",
            visibility: str = 'Private', max_imbalance: float = 3.0,
            nmax: int = 400, epochs: int = 3, freeze_encoder: bool = True,
            batch_size: int = 8, model_type: str = 'mpnet',
            model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        """
        on button click create with option: from category name
        """

        self.setup()
        self.load_corpus(corpus_name)
        self.get_labels_by_topics()
        self.train_PUmodel(
            max_imbalance, nmax, epochs, freeze_encoder=freeze_encoder,
            batch_size=batch_size, model_type=model_type,
            model_name=model_name)
        self.DM.save_model_json(
            self.project_folder, description, visibility, tag, 'Keyword-based',
            self.dc.config)

    def on_retrain(self, epochs: int = 3):
        """
        on button click retrain
        """
        self.retrain_model(epochs)

    def on_classify(self):
        """
        on button click classify
        """
        self.inference()

    def on_evaluate(self, true_label_name: str):
        """
        on button click evaluate
        """
        self.evaluate_PUlabels(true_label_name)

    def on_sample(self, sampler: str = "", n_samples: int = -1):
        """
        on button click sample

        Parameters
        ----------
        sampler : str, optional (default = "")
            Type of sampler. If "", the sampler is read from the global
            parameters
        n_samples : int, optional (default=-1)
            Number of samples to return.
            If -1, the number of samples is taken from the configuration file
        """
        self.sample_documents(sampler, fmt="json", n_samples=n_samples)

    def on_save_feedback(self):
        """
        on button click save feedback
        """
        self.annotate(fmt='json')
