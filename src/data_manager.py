"""
Defines a data manager class to provide basic read-write functionality for
the project.

@author: J. Cid-Sueiro, A. Gallardo-Antolin, T. Ahlers
"""

import pathlib
import pandas as pd
import numpy as np
import scipy.sparse as scp
import logging
import shutil
import yaml

from time import time
from sklearn.preprocessing import normalize
from langdetect import detect

import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# import gc


def detect_english(x):
    """
    Returns True is x contains text in English.

    Parameters
    ----------
    x : str
        Input string

    Returns
    -------
    True if x contains English text, False otherwise.
    """
    # Default output value. Only if langdetect detects english, y will be True
    y = False

    # A try-except is required because detect() raises execution errors for
    # invalid strings that may appear for some inputs x.
    try:
        # Check if the string contains at least one alphabetic character
        # (otherwise, the lang detector raises an error)
        if x.lower().islower():
            y = detect(x) == 'en'
    except Exception:
        logging.warning(f"-- Language detection error in string {x}")

    return y


class DataManager(object):
    """
    This class contains all read / write functionalities required by the
    domain_classification project.

    It assumes that source and destination data will be stored in files.
    """

    def __init__(self, path2source, path2datasets, path2models,
                 path2embeddings=None):
        """
        Initializes the data manager object

        Parameters
        ----------
        path2source: str or pathlib.Path
            Path to the folder containing all external source data
        path2datasets: str or pathlib.Path
            Path to the folder containing datasets
        path2models: str or pathlib.Path
            Path to the folder containing classifier models
        path2embeddings: str or pathlib.Path
            Path to the folder containing the document embeddings
        """

        self.path2source = pathlib.Path(path2source)
        # self.path2labels = pathlib.Path(path2labels)   # No longer used
        self.path2datasets = pathlib.Path(path2datasets)
        self.path2models = pathlib.Path(path2models)

        if path2embeddings is not None:
            self.path2embeddings = pathlib.Path(path2embeddings)

        # The path to the corpus is given when calling the load_corpus method
        self.path2corpus = None

        # Default name of the corpus. It can be changed by the load_corpus or
        # the load_dataset methods
        self.corpus_name = ""

        # Dictionary of metadata
        self.metadata = None

        return

    def _get_path2feather(self, sampling_factor):
        """
        Returns the path to the feather file associated to the subcorpubs based
        on the given sampling factor

        Parameters
        ----------
        sampling_factor : int
            Sampling factor
        """

        if sampling_factor == 1:
            path2feather = self.path2corpus / 'corpus' / 'corpus.feather'
        else:
            # Generate label (docs per million)
            dpm = str(int(sampling_factor * 1e6))
            path2feather = (self.path2corpus / 'corpus'
                            / f'corpus_{dpm}.feather')

        return path2feather

    def __load_metadata(self):
        """
        Loads the metadata file if it exists. If not, self.metadata takes the
        default value (None)
        """

        path2metadata = self.path2corpus / 'metadata.yaml'
        if path2metadata.is_file():
            with open(path2metadata, 'r', encoding='utf8') as f:
                self.metadata = yaml.safe_load(f)
        else:
            self.metadata = None

        return

    def __update_metadata(self):

        path2metadata = self.path2corpus / 'metadata.yaml'
        with open(path2metadata, 'w') as f:
            yaml.dump(self.metadata, f)

        return

    def __determineCorpusHasEmbeddings(self, columns):

        self.metadata["corpus_has_embeddings"] = bool(
            (np.array(columns) == 'embeddings').sum() > 0)
        self.__update_metadata()

        return

    def get_metadata(self):
        """
        A method to access the metadata attribute from outside the data_manager
        """
        return self.metadata

    def is_model(self, class_name):
        """
        Checks if a model exist for the given class_name in the folder of
        models

        Parameters
        ----------
        class_name : str
            Name of the class

        Returns
        -------
            True if the model folder exists
        """

        path2model = self.path2models / class_name

        return path2model.is_dir()

    def get_corpus_list(self):
        """
        Returns the list of available corpus
        """

        corpus_list = [e.name for e in self.path2source.iterdir()
                       if e.is_dir()]

        return corpus_list

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

    def get_annotation_list(self):
        """
        Returns the list of available corpus
        """

        # Data location
        prefix = f"labels_{self.corpus_name}_"
        folder = self.path2corpus / 'annotations'

        # If there is no annotation folder, return an empty list.
        if not folder.is_dir():
            return []

        # Read files
        files = [e.stem for e in folder.iterdir()
                 if e.is_file() and e.stem.startswith(prefix)]

        # Remove prefixes and get tags only:
        tags = [e[len(prefix):] for e in files]

        # Since the dataset folder might contain diferent file formats of the
        # same annotation file, we take unique names
        tags = list(set(tags))

        return tags

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

    def load_corpus(self, corpus_name, sampling_factor=1):
        """
        Loads a dataframe of documents from a given corpus.

        When available, the names of the relevant dataframe components are
        mapped to normalized names: id, title, description, keywords and
        target_xxx

        Parameters
        ----------
        corpus_name : str
            Name of the corpus. It should be the name of a folder in
            self.path2source

        sampling_factor : float, optional (default=1)
            Fraction of documents to be taken from the original corpus.
            (Used for SemanticScholar and Patstat only)
        """

        def corpus_has_embeddings(path2texts):

            try:
                pd.read_parquet(next(path2texts.glob('**/*')),
                                columns=['embeddings'])
                return True
            except Exception:
                return False

        def sample_sub_set_from_folder(path2texts, selected_cols,
                                       sampling_factor):

            fpaths = np.array([f for f in path2texts.glob('**/*')
                               if f.is_file() and f.suffix == '.parquet'])
            read_count = np.round(len(fpaths) * sampling_factor).astype(int)
            rand_idx = np.random.choice(
                len(fpaths), read_count, replace=False).astype(int)

            df_corpus = pd.DataFrame([])
            for k, path_k in enumerate(fpaths[rand_idx]):
                print(f"-- -- Loading file {k + 1} out of {len(rand_idx)},"
                      " Coprus len: {len(df_corpus)} \r", end="")
                dfk = pd.read_parquet(path_k, columns=selected_cols)
                df_corpus = pd.concat([df_corpus, dfk])

            return df_corpus

        # Loading corpus
        logging.info(f'-- Loading corpus {corpus_name}')
        t0 = time()

        self.corpus_name = corpus_name
        self.path2corpus = self.path2source / corpus_name
        self.__load_metadata()
        path2feather = self._get_path2feather(sampling_factor)

        # #################################################
        # Load corpus data from feather file (if it exists)
        if path2feather.is_file():

            logging.info(f'-- -- Feather file {path2feather} found...')
            df_corpus = pd.read_feather(path2feather)

            # Log results
            logging.info(f"-- -- Corpus {corpus_name} with {len(df_corpus)} "
                         f" documents loaded in {time() - t0:.2f} secs.")

            self.__determineCorpusHasEmbeddings(df_corpus.columns)

            return df_corpus

        # #########################################
        # Load corpus data from its original source

        # By default, neither corpus cleaning nor language filtering are done
        clean_corpus = corpus_name in {
            'SemanticScholar', 'SemanticScholar_emb',
            'patstat', 'patstat_emb',
            'AEI_projects', 'CORDIS.parquet', 'S2CS.parquet'}
        remove_non_en = corpus_name in {
            'SemanticScholar', 'SemanticScholar_emb',
            'patstat', 'patstat_emb'}

        if corpus_name == 'EU_projects':

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
            # df_corpus = df_corpus1.append(df_corpus2, ignore_index=True)
            df_corpus = pd.concat([df_corpus1, df_corpus2])
            df_corpus = df_corpus[['acronym', 'title', 'objective', 'rcn']]

            # Map column names to normalized names
            mapping = {'rcn': 'id',
                       'objective': 'description'}
            df_corpus.rename(columns=mapping, inplace=True)

            # Fill nan cells with empty strings
            df_corpus.fillna("", inplace=True)

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
            logging.info(f'-- -- Raw corpus {corpus_name} read with '
                         f'{len(df_corpus)} documents')

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

        elif corpus_name == 'CORDIS.parquet':

            # ####################
            # Paths and file names

            path2texts = self.path2corpus / 'corpus'
            fpaths = [f for f in path2texts.glob('**/*')
                      if f.is_file() and f.suffix == '.parquet']
            n_files = len(fpaths)

            # ###########
            # Load corpus

            for k, path_k in enumerate(fpaths):
                print(f"-- -- Loading file {k + 1} out of {n_files}  \r",
                      end="")
                dfk = pd.read_parquet(path_k)

                # Original fields are:
                #   'id', 'title', 'objective', 'startDate',
                #   'ecMaxContribution', 'euroSciVocCode', 'rawtext', 'lemmas'
                dfk = dfk[['id', 'title', 'objective', 'euroSciVocCode']]

                if k == 0:
                    df_corpus = dfk
                else:
                    # df_corpus0 = df_corpus.append(dfk, ignore_index=True)
                    df_corpus = pd.concat([df_corpus, dfk])

            logging.info(f'-- -- Raw corpus {corpus_name} read with '
                         f'{len(dfk)} documents')

            # Map column names to normalized names
            # We use "Referencia", renamed as "id", as the project id.
            mapping = {'objective': 'description'}
            df_corpus.rename(columns=mapping, inplace=True)

            # Map list of euroSciVoc codes to a string (otherwise, no
            # feather file can be saved)
            col = 'euroSciVocCode'   # Just to abbreviate
            if col in df_corpus:
                df_corpus[col] = df_corpus[col].apply(
                    lambda x: ','.join(
                        x.astype(str)) if x is not None else '')

        elif corpus_name == 'S2CS.parquet':

            # ####################
            # Paths and file names

            path2texts = self.path2corpus / 'corpus'
            fpaths = [f for f in path2texts.glob('**/*')
                      if f.is_file() and f.suffix == '.parquet']
            n_files = len(fpaths)

            # ###########
            # Load corpus

            for k, path_k in enumerate(fpaths):
                print(f"-- -- Loading file {k + 1} out of {n_files}  \r",
                      end="")
                dfk = pd.read_parquet(path_k)

                # Original fields are:
                #   'id', 'title', 'paperAbstract', 'doi', 'year',
                #   'fieldsOfStudy', 'rawtext', 'lemmas'
                dfk = dfk[['id', 'title', 'paperAbstract', 'fieldsOfStudy']]

                if k == 0:
                    df_corpus = dfk
                else:
                    # df_corpus = df_corpus.append(dfk, ignore_index=True)
                    df_corpus = pd.concat([df_corpus, dfk])

            logging.info(f'-- -- Raw corpus {corpus_name} read with '
                         f'{len(dfk)} documents')

            # Map column names to normalized names
            mapping = {'paperAbstract': 'description',
                       'fieldsOfStudy': 'keywords'}

            df_corpus.rename(columns=mapping, inplace=True)

        elif corpus_name in {'SemanticScholar', 'SemanticScholar_emb'}:
            path2metadata = self.path2corpus / 'metadata.yaml'

            if not path2metadata.is_file():
                logging.error(
                    f"-- A metadata file in {path2metadata} is missed. It is "
                    "required for this corpus. Corpus not loaded")

            with open(path2metadata, 'r', encoding='utf8') as f:
                metadata = yaml.safe_load(f)
            path2texts = pathlib.Path(metadata['corpus'])

            if not corpus_has_embeddings(path2texts):
                df = dd.read_parquet(path2texts, split_row_groups=True)
                dfsmall = df.sample(frac=sampling_factor, random_state=0)

                with ProgressBar():
                    df_corpus = dfsmall.compute()

                # Remove unrelevant fields
                selected_cols = ['id', 'title',
                                 'paperAbstract', 'fieldsOfStudy']
                if 'embeddings' in df_corpus:
                    selected_cols += ['embeddings']
                df_corpus = df_corpus[selected_cols]
            else:
                selected_cols = np.array(['id', 'title', 'paperAbstract', 'fieldsOfStudy', 'embeddings'])
                df_corpus = sample_sub_set_from_folder(path2texts, selected_cols, sampling_factor)
                logging.info(f'-- -- Raw corpus {corpus_name} read with '
                             f'{len(df_corpus)} documents')

            # Map column names to normalized names
            mapping = {'paperAbstract': 'description',
                       'fieldsOfStudy': 'keywords'}
            df_corpus.rename(columns=mapping, inplace=True)

            # Map list of keywords to a string (otherwise, no feather file can
            # be saved)
            col = 'keywords'   # Just to abbreviate
            df_corpus[col] = df_corpus[col].apply(
                lambda x: ','.join(x.astype(str)) if x is not None else '')
        elif corpus_name in {'patstat', 'patstat_emb'}:

            path2metadata = self.path2corpus / 'metadata.yaml'

            if not path2metadata.is_file():
                logging.error(
                    f"-- A metadata file in {path2metadata} is missed. It is "
                    "required for this corpus. Corpus not loaded")

            with open(path2metadata, 'r', encoding='utf8') as f:
                metadata = yaml.safe_load(f)
            path2texts = pathlib.Path(metadata['corpus'])

            df = dd.read_parquet(path2texts)
            dfsmall = df.sample(frac=sampling_factor, random_state=0)

            with ProgressBar():
                df_corpus = dfsmall.compute()

            # Remove unrelevant fields
            # Available fields are: appln_id, docdb_family_id, appln_title,
            # appln_abstract, appln_filing_year, earliest_filing_year,
            # granted, appln_auth, receiving_office, ipr_type
            selected_cols = ['appln_id', 'appln_title', 'appln_abstract']
            if 'embeddings' in df_corpus:
                selected_cols += ['embeddings']
            df_corpus = df_corpus[selected_cols]

            # Map column names to normalized names
            mapping = {'appln_id': 'id',
                       'appln_title': 'title',
                       'appln_abstract': 'description'}
            df_corpus.rename(columns=mapping, inplace=True)

        else:
            logging.warning("-- Unknown corpus")
            return None

        # ############
        # Clean corpus
        if clean_corpus:

            l0 = len(df_corpus)
            logging.info(f"-- -- {l0} base documents loaded")

            # Remove duplicates, if any
            df_corpus.drop_duplicates(subset=['id'], inplace=True)
            l1 = len(df_corpus)
            logging.info(f"-- -- {l0 - l1} duplicated documents removed")

            # Remove documents with missing data, if any
            ind_notna = df_corpus['title'].notna()
            df_corpus = df_corpus[ind_notna]
            ind_notna = df_corpus['description'] == 0
            df_corpus = df_corpus[~ind_notna]

            # Fill nan cells with empty strings
            df_corpus.fillna("", inplace=True)

            # Remove documents with zero-length title or description
            df_corpus = df_corpus[df_corpus['title'] != ""]
            df_corpus = df_corpus[df_corpus['description'] != ""]

            # Remove special characters
            df_corpus['title'] = df_corpus['title'].str.replace('\t', '')
            df_corpus['description'] = (
                df_corpus['description'].str.replace('\t', ''))

            # Log results
            l2 = len(df_corpus)
            logging.info(f"-- -- {l1 - l2} documents with empty title or "
                         "description: removed")

        # ###############################################
        # Remove documents without description in english
        # Note that if save_feather is False, non-english removal mu
        if remove_non_en:

            l0 = len(df_corpus)
            logging.info("-- -- Applying language filter. This may take a "
                         "while")

            # if 1 == 2:
            #    df_corpus['eng'] = (
            #        df_corpus['title'] + ' ' + df_corpus['description']).apply(
            #            detect_english)
            # else:
            iPercent = -1
            df_corpus['eng'] = False
            for count, (index, row) in enumerate(df_corpus.iterrows()):
                df_corpus.loc[index, 'eng'] = detect_english(
                    row['title'] + ' ' + row['description'])
                if int(100*count/len(df_corpus)) > iPercent:
                    iPercent = int(100*count/len(df_corpus))
                    logging.info(f"-- -- {iPercent} % of documents processed for applying language filter")

            df_corpus = df_corpus[df_corpus['eng']]
            df_corpus.drop(columns='eng', inplace=True)

            # Log results
            l1 = len(df_corpus)
            logging.info(f"-- -- {l0 - l1} non-English documents: removed")

        # Reset the index and drop the old index
        df_corpus = df_corpus.reset_index(drop=True)

        self.__determineCorpusHasEmbeddings(df_corpus.columns)

        # ############
        # Log and save

        # Save to feather file
        logging.info(f"-- -- Corpus {corpus_name} with {len(df_corpus)} "
                     f" documents loaded in {time() - t0:.2f} secs.")
        df_corpus.to_feather(path2feather)
        logging.info(f"-- -- Corpus saved in feather file {path2feather}")

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

    def load_topic_metadata(self):
        """
        Loads the metadata associated to the topic matrix from the selected
        corpus
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
            return None, None

        data = np.load(path2topics)

        # Extract topic descriptions
        topic_words = data['descriptions']

        # Load metadata
        path2metadata = self.path2corpus / topic_folder / metadata_fname
        df_metadata = pd.read_csv(path2metadata)
        logging.info(f'-- Topic matrix metadata loaded about '
                     f'{len(df_metadata)} documents')

        return df_metadata, topic_words

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

    def reset_labels(self, tag=""):
        """
        Delete all files related to a given class

        Parameters
        ----------
        tag : str, optional (default="")
            Name of the class to be removed
        """

        # Remove csv file
        # fname = f"labels_{self.corpus_name}_{tag}.csv"
        # path2labelset = self.path2labels / fname
        # path2labelset.unlink()

        # Remove dataset
        fstem = f"dataset_{self.corpus_name}_{tag}"
        for p in self.path2datasets.glob(f"{fstem}.*"):
            p.unlink()

        # Remove model
        path2model = self.path2models / tag
        if path2model.is_dir():
            shutil.rmtree(path2model)

        logging.info(f"-- -- Labels {tag} removed")

    def import_AI_subcorpus(self, ids_corpus=None, tag="imported"):
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
        ids_pos: list
            List of ids of documents from the positive class
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

        # Only the ids of the docs with positive labels are returned
        ids_pos = df_labels.id.tolist()

        # This is for backward compatibility only
        msg = ""

        # The log message is returned to be shown in a GUI, if needed
        return ids_pos, msg

    def load_selected_docs(self, tag=""):
        """
        Loads a temporal dataframe of documents selected for annotation

        Parameters
        ----------
        tag : str, optional (default="")
            Label suffix for the file name

        Returns
        -------
        df : pandas.DataFrame
            Selected documents
        """

        # Create temporary folder
        fpath = self.path2datasets / "temp" / f"selected_docs_{tag}.csv"
        # Below, index_col=0 is used to use the first column as indices.
        # Otherwise, an additional index colums would be used.
        df = pd.read_csv(fpath, index_col=0)

        return df

    def save_selected_docs(self, df_docs, tag=""):
        """
        Save a temporal dataframe of documents selected for annotation

        Parameters
        ----------
        df_docs : pandas.DataFrame
            Dataframe of documents to be annotated
        tag : str, optional (default="")
            Suffix for the file name
        """

        # Create temporary folder
        folder = self.path2datasets / "temp"
        if not folder.exists():
            folder.mkdir()

        # Save
        fpath = folder / f"selected_docs_{tag}.csv"
        # Below, index_col=0 is used to use the first column as indices.
        # Otherwise, an additional index colums would be used.
        df_docs.to_csv(fpath)

        return

    def load_new_labels(self, tag=""):
        """
        Loads a temporal dataframe of documents selected for annotation

        Parameters
        ----------
        tag : str, optional (default="")
            Suffix for the file name

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of labels
        """

        # Create temporary folder
        fpath = self.path2datasets / "temp" / f"new_labels_{tag}.csv"
        df = pd.read_csv(fpath, index_col=0)

        return df

    def save_new_labels(self, idx, labels, tag=""):
        """
        Save labels from the last annotation round in a temporary file

        Parameters
        ----------
        idx : list like
            Indices of the annotated documents.
        labels : list
            List of labels
        tag : str, optional (default="")
            Label suffix for the file name
        """

        # Make dataframe (to save easily into a csv file)
        df_labels = pd.DataFrame(labels, index=idx, columns=['labels'])

        # Create temporary folder
        folder = self.path2datasets / "temp"
        if not folder.exists():
            folder.mkdir()

        # Save
        fpath = folder / f"new_labels_{tag}.csv"
        df_labels.to_csv(fpath)

        return

    def remove_temp_files(self, tag):
        """
        Removes temporary files associated to a given annotation round
        """

        folder = self.path2datasets / "temp"
        # Remove selected docs file.
        fpath = folder / f"selected_docs_{tag}.csv"
        if fpath.exists():
            fpath.unlink()
        # Remove labels file.
        fpath = folder / f"new_labels_{tag}.csv"
        if fpath.exists():
            fpath.unlink()

        return

    def import_annotations(self, tag):
        """
        Loads a file with annotations.

        Parameters
        ----------
        tag: str
            Name used to identify the annmotation file.

        Returns
        -------
        df_annotations: pandas.dataFrame
            Dataframe of annotations.
        """

        # Read labels file
        path2labels = (self.path2corpus / 'annotations'
                       / f"labels_{self.corpus_name}_{tag}.csv")
        df_annotations = pd.read_csv(path2labels)

        return df_annotations

    def export_annotations(self, df_annotations, tag):
        """
        Export a dataframe of annotations to csv file.

        Parameters
        ----------
        df_annotations: pandas.dataFrame
            A dataframse of annotations
        tag: str, optional (default="imported")
            Name for the domain. To be included as a suffix of the file name.

        """

        # Create folder if it does not exist
        folder = self.path2corpus / 'annotations'
        if not folder.is_dir():
            folder.mkdir()

        # Export annotations
        path2labels = folder / f"labels_{self.corpus_name}_{tag}.csv"
        df_annotations.to_csv(path2labels)

        return

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
        # deep copy to prevent that embeddings are deleted in object
        df_dataset = df_dataset.copy(deep=True)
        df_dataset.drop(['embeddings'], axis=1, errors='ignore', inplace=True)

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
