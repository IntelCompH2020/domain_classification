"""
Defines classes and methods providing the main functionality for document
selection through keywords, a category name or a weighted list of topics.

@author: J. Cid-Sueiro, A. Gallardo-Antolin
"""

import numpy as np
import pandas as pd
import pathlib
import pickle
import logging

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    pipeline, AutoModelForSequenceClassification, AutoTokenizer)

# Some libraries required for evaluation
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, roc_curve


class CorpusProcessor(object):
    """
    A container of corpus preprocessing methods
    It provides basic processing methods to a corpus of text documents
    The input corpus must be given by a list of strings (or a pandas series
    of strings)
    """

    def __init__(self, path2embeddings=None, path2zeroshot=None):
        """
        Initializes a preprocessor object

        Parameters
        ----------
        path2embeddings : str or pathlib.Path or None, optional (default=None)
            Path to the folder containing the document embeddings.
            If None, no embeddings will be used. Document scores will be based
            in word counts
        path2zeroshot : str or pathlib.Path or None, optional (default=None)
            Path to the folder containing the pretrained zero-shot model
            If None, zero-shot classification will not be available.
        """

        if path2embeddings is not None:
            self.path2embeddings = pathlib.Path(path2embeddings)
        else:
            self.path2embedding = None
            logging.info("-- No path to embeddings. "
                         "Document scores will be based in word counts")

        if path2zeroshot is not None:
            self.path2zeroshot = pathlib.Path(path2zeroshot)
        else:
            self.path2zeroshot = None
            logging.info("-- No path to zero-shot model. "
                         "Zero-shot classification not available")

        logging.info("-- Preprocessor object created.")

        return

    def score_docs_by_keyword_count(self, corpus, keywords):
        """
        Computes a score for every document in a given pandas dataframe
        according to the frequency of appearing some given keywords

        Parameters
        ----------
        corpus : list (or pandas.Series) of str
            Input corpus.
        keywords : list of str
            List of keywords

        Returns
        -------
        score : list of float
            List of scores, one per document in corpus
        """

        score = []
        n_docs = len(corpus)

        for i, doc in enumerate(corpus):
            print(f"-- Processing document {i} out of {n_docs}   \r", end="")
            reps = [doc.count(k) for k in keywords]
            score.append(sum(reps))

        return score

    def score_docs_by_keywords(self, corpus, keywords,
                               model_name='all-MiniLM-L6-v2'):
        """
        Computes a score for every document in a given pandas dataframe
        according to the frequency of appearing some given keywords

        Parameters
        ----------
        corpus : list (or pandas.Series) of str
            Input corpus.
        keywords : list of str
            List of keywords
        model_name : str, optinal (default = 'all-MiniLM-L6-v2')
            Name of the SBERT transformer model

        Returns
        -------
        score : list of float
            List of scores, one per document in corpus
        """

        # Check if embeddings have been provided
        if self.path2embeddings is None:
            score = self.score_docs_by_keyword_count(corpus, keywords)
            return score

        # 1. Load sentence transformer model
        model = SentenceTransformer(model_name)

        # 2. Document embeddings. If precalculated, load the embeddings.
        # Otherwise, compute the embeddings.
        embeddings_out_fname = f'corpus_embed_{model_name}.pkl'
        embeddings_fname = self.path2embeddings / embeddings_out_fname

        if pathlib.Path.exists(embeddings_fname):
            # Load embedding of the corpus documents
            with open(embeddings_fname, "rb") as f_in:
                doc_embeddings = pickle.load(f_in)

            logging.info(f'-- Corpus {embeddings_fname} loaded with '
                         f'{len(doc_embeddings)} doc embeddings')
        else:
            # Corpus embedding computation
            logging.info(f'-- No embedding file {embeddings_fname} available')
            logging.info(f'-- Embedding docs with model {model_name}')
            n_docs = len(corpus)
            # n_docs = 2000
            batch_size = 128
            doc_embeddings = model.encode(corpus.values[0:n_docs],
                                          batch_size=batch_size)

            # Saving doc embedding
            with open(embeddings_fname, "wb") as fOut:
                pickle.dump(doc_embeddings, fOut)
            logging.info(f'-- Corpus embeddings saved in {embeddings_fname}')

        # 3. Keyword embeddings
        logging.info(f'-- Embedding keywords with model {model_name}')
        keyword_embeddings = model.encode(keywords)

        # 4. Cosine similarities computation
        distances = cosine_similarity(doc_embeddings, keyword_embeddings)
        score = np.mean(distances, axis=1)

        return score

    def score_docs_by_zeroshot(self, corpus, keyword):
        """
        Computes a score for every document in a given pandas dataframe
        according to a given keyword and a pre-trained zero-shot classifier

        Parameters
        ----------
        corpus : list (or pandas.Series) of str
            Input corpus.
        keyword : str
            Keyword defining the target category

        Returns
        -------
        score : list of float
            List of scores, one per document in corpus

        Notes
        -----
        Adapted from code contributed by BSC for the IntelComp project.
        """

        # Check if there is a zero-shot model
        if self.path2zeroshot is None:
            logging.warning("-- -- No zero-shot model is available.")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.path2zeroshot)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.path2zeroshot)

        # Use device=0 for GPU inference
        zsc = pipeline("zero-shot-classification", model=model,
                       tokenizer=tokenizer, device=-1)

        score = []
        n_docs = len(corpus)

        # Initialize counter of the number of document truncations required
        # by the zero-shot model
        n_trunc = 0
        # Initialize metric of the minimum size of a reduced document
        li_min = 1e10

        for i, doc in enumerate(corpus):
            li = len(doc)
            print(f"-- Processing document {i} / {n_docs}, size {li} \r",
                  end="")

            in_process = True
            while in_process:
                try:
                    # Zero-shot classification for doc_i and
                    # category keyboard
                    doc_i = doc[:li]
                    result = zsc(doc_i, keyword, multi_label=False)
                    in_process = False

                except:
                    # An error is raised because "The expanded size of the
                    # tensor (519) must match the existing size (514) at
                    # non-singleton dimension 1.  Target sizes: [1, 519].
                    # Tensor sizes: [1, 514]"
                    # We have to truncate the document.
                    li = 3 * li // 4
                    li_min = min(li_min, li)
                    n_trunc += 1

            # Save score
            score_i = sum(result['scores'])
            score.append(score_i)

        logging.info(f"-- -- A document truncation was required {n_trunc} "
                     f"times, up to a size of at least {li_min} characters")

        return score

    def compute_keyword_stats(self, corpus, keywords):
        """
        Computes keyword statistics

        Parameters
        ----------
        corpus : list (or pandas.Series) of str
            Input corpus.
        keywords : list of str
            List of keywords

        Returns
        -------
        df_stats : dict
            Dictionary of document frequencies per keyword
            df_stats[k] is the number of docs containing keyword k
        kf_stats : dict
            Dictionary of keyword frequencies
            df_stats[k] is the number of times keyword k appers in the corpus
        """

        n_keywords = len(keywords)
        df_stats, kf_stats = {}, {}

        for i, k in enumerate(keywords):
            print(f"-- Processing keyword {i + 1} out of {n_keywords}    \r",
                  end="")
            counts = [doc.count(k) for doc in corpus]
            df_stats[k] = np.count_nonzero(counts)
            kf_stats[k] = np.sum(counts)

        return df_stats, kf_stats

    def get_top_scores(self, scores, n_max=1e100, s_min=0):
        """
        Select the elements from a given list of numbers that fulfill some
        conditions

        Parameters
        ----------
        n_max: int or None, optional (defaul=1e100)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=0)
            Minimum score. Only elements strictly above s_min are selected
        """

        # Make sure that the score values are in a numpy array
        s = np.array(scores)

        # n_max cannot be higher that the size of the array of scores
        n_max = min(n_max, len(scores))

        isort = np.argsort(-s)
        isort = isort[: n_max]

        ind = [i for i in isort if s[i] > s_min]

        return ind

    def performance_metrics(self, scores, target, s_min, n_max):
        """
        Compute evaluation metrics for the generation of the subcorpus using
        a keyword

        To do so, it requires from self.df_corpus to have at least the
        following columns: id, title, description, target_bio, target_tic,
        target_ene

        Parameters
        ----------
        scores : np.array
            Score values
        target : np.array
            Target values
        s_min: float, optional (default=0)¡
            Minimum score. Only elements strictly above s_min are selected
        n_max: int or None, optional (defaul=1e100)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no limit

        Returns
        -------
        eval_scores: dict
            A dictionary of evaluation metrics.
        """

        # Sort scores and target values
        s = np.array(scores)
        ssort = -np.sort(-s)
        isort = np.argsort(-s)
        target_sorted = target[isort]

        # Fill the information related to the retrieved docs
        df_output_smin = np.multiply(ssort > s_min, 1)
        df_output_nmax = np.hstack(
            (np.ones(n_max), np.zeros(len(s) - n_max)))

        # Metrics computation at s_min threshold
        tn, fp, fn, tp = confusion_matrix(
            target_sorted, df_output_smin).ravel()
        tpr_smin = tp / (tp + fn)
        fpr_smin = fp / (fp + tn)
        acc_smin = (tp + tn) / (tp + tn + fp + fn)
        bal_acc_smin = 0.5 * (tpr_smin + 1 - fpr_smin)

        # ## prfs = precision_recall_fscore_support(
        # ##     df_retrieved_docs['target'],
        # ##     df_retrieved_docs['output_smin'])
        # ## ap = average_precision_score(df_retrieved_docs['target'],
        # ##                             df_retrieved_docs['score'])

        # Metrics computation at n_max retrieved documents
        tn, fp, fn, tp = confusion_matrix(
            target_sorted, df_output_nmax).ravel()
        tpr_nmax = tp / (tp + fn)
        fpr_nmax = fp / (fp + tn)
        acc_nmax = (tp + tn) / (tp + tn + fp + fn)
        bal_acc_nmax = 0.5 * (tpr_nmax + 1 - fpr_nmax)

        # Precision-Recall curve
        # ##precision, recall, thresholds = precision_recall_curve(
        # ##    df_retrieved_docs['target'], df_retrieved_docs['score'])

        # ROC curve
        fpr_roc, tpr_roc, thresholds = roc_curve(target_sorted, ssort)

        # Dictionary with the evaluation results
        tpr_roc_float = [float(k) for k in tpr_roc]
        fpr_roc_float = [float(k) for k in fpr_roc]
        eval_scores = {
            'acc_smin': float(acc_smin),
            'bal_acc_smin': float(bal_acc_smin),
            'tpr_smin': float(tpr_smin),
            'fpr_smin': float(fpr_smin),
            'acc_nmax': float(acc_nmax),
            'bal_acc_nmax': float(bal_acc_nmax),
            'tpr_nmax': float(tpr_nmax),
            'fpr_nmax': float(fpr_nmax),
            'tpr_roc': tpr_roc_float,
            'fpr_roc': fpr_roc_float}

        return eval_scores


class CorpusDFProcessor(object):
    """
    A container of corpus processing methods.
    It assumes that a corpus is given by a dataframe of documents.

    Each dataframe must contain three columns:
    id: document identifiers
    title: document titles
    description: body of the document text
    """

    def __init__(self, df_corpus, path2embeddings=None, path2zeroshot=None):
        """
        Initializes a preprocessor object

        Parameters
        ----------
        df_corpus : pandas.dataFrame
            Input corpus.
        path2embeddings : str or pathlib.Path or None, optional (default=None)
            Path to the folder containing the document embeddings.
            If None, no embeddings will be used. Document scores will be based
            in word counts
        path2zeroshot : str or pathlib.Path or None, optional (default=None)
            Path to the folder containing the pretrained zero-shot model
            If None, zero-shot classification will not be available.
        """

        if path2embeddings is not None:
            self.path2embeddings = pathlib.Path(path2embeddings)
        else:
            self.path2embedding = None
            logging.info("-- No path to embeddings. "
                         "Document scores will be based on word counts")

        if path2zeroshot is not None:
            self.path2zeroshot = pathlib.Path(path2zeroshot)
        else:
            self.path2zeroshot = None
            logging.info("-- No path to zero-shot model. "
                         "Zero-shot classification not available")

        self.df_corpus = df_corpus

        # This class uses methods from the corpus processor class.
        self.prep = CorpusProcessor(self.path2embeddings, self.path2zeroshot)

        return

    def remove_docs_from_topics(self, T, df_metadata, col_id='id'):
        """
        Removes, from a given topic-document matrix and its corresponding
        metadata dataframe, all documents that do not belong to the corpus

        Parameters
        ----------
        T: numpy.ndarray or scipy.sparse
            Topic matrix (one column per topic)
        df_metadata: pandas.DataFrame
            Dataframe of metadata. It must include a column with document ids
        col_id: str, optional (default='id')
            Name of the column containing the document ids in df_metadata

        Returns
        -------
        T_out: numpy.ndarray or scipy.sparse
            Reduced topic matrix (after document removal)
        df_out: pands.DataFrame
            Metadata dataframe, after document removal
        """

        # Find doc ids in df_metadats that exist in the corpus dataframe
        corpus_ids = self.df_corpus['id']
        detected_ids = df_metadata[col_id].isin(corpus_ids)

        # Filter out strange document ids
        T_out = T[detected_ids]
        df_out = df_metadata[detected_ids]

        return T_out, df_out

    def compute_keyword_stats(self, keywords, wt=2):
        """
        Computes keyword statistics

        Parameters
        ----------
        corpus : dataframe
            Dataframe of corpus.
        keywords : list of str
            List of keywords

        Returns
        -------
        df_stats : dict
            Dictionary of document frequencies per keyword
            df_stats[k] is the number of docs containing keyword k
        kf_stats : dict
            Dictionary of keyword frequencies
            df_stats[k] is the number of times keyword k appers in the corpus
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
        """

        # We take the (closest) integer part only
        intwt = round(wt)

        corpus = ((self.df_corpus['title'] + ' ') * intwt
                  + self.df_corpus['description'])

        df_stats, kf_stats = self.prep.compute_keyword_stats(corpus, keywords)

        return df_stats, kf_stats

    def score_by_keyword_count(self, keywords, wt=2):
        """
        Computes a score for every document in a given pandas dataframe
        according to the frequency of appearing some given keywords

        Parameters
        ----------
        corpus : dataframe
            Dataframe of corpus.
        keywords : list of str
            List of keywords
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor

        Returns
        -------
        score : list of float
            List of scores, one per documents in corpus
        """

        score_title = self.prep.score_docs_by_keyword_count(
            self.df_corpus['title'], keywords)
        score_descr = self.prep.score_docs_by_keyword_count(
            self.df_corpus['description'], keywords)

        score = wt * np.array(score_title) + np.array(score_descr)

        return score

    def score_by_keywords(self, keywords, wt=2, model_name='all-MiniLM-L6-v2',
                          method='embedding'):
        """
        Computes a score for every document in a given pandas dataframe
        according to the frequency of appearing some given keywords

        Parameters
        ----------
        keywords : list of str
            List of keywords
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
            This input argument is used if self.path2embeddings is None only
        model_name : str, optinal (default = 'all-MiniLM-L6-v2')
            Name of the SBERT transformer model
        method : str in {'embedding', 'count'}
            - If 'count', documents are scored according to word counts
            - If 'embedding', scores are based on neural embeddings

        Returns
        -------
        score : list of float
            List of scores, one per documents in corpus
        """

        # Check if embeddings have been provided
        if method == 'count' or self.path2embeddings is None:
            score = self.score_by_keyword_count(keywords, wt)
            return score

        # Copy relevant columns only
        df_dataset = self.df_corpus.loc[:, ['id', 'title', 'description']]

        # Join title and description into a single column
        df_dataset.loc[:, 'text'] = (df_dataset['title'] + '. '
                                     + df_dataset['description'])

        df_dataset.drop(columns=['description', 'title'], inplace=True)

        score = self.prep.score_docs_by_keywords(
            df_dataset['text'], keywords, model_name)

        return score

    def score_by_zeroshot(self, keyword):
        """
        Computes a score for every document in a given pandas dataframe
        according to the relevance of a given keyword according to a pretrained
        zero-shot classifier

        Parameters
        ----------
        keyword : str
            Keywords defining the target category

        Returns
        -------
        score : list of float
            List of scores, one per documents in corpus
        """

        # Copy relevant columns only
        df_dataset = self.df_corpus.loc[:, ['id', 'title', 'description']]

        # Join title and description into a single column
        df_dataset.loc[:, 'text'] = (df_dataset['title'] + '. '
                                     + df_dataset['description'])
        df_dataset.drop(columns=['description', 'title'], inplace=True)

        score = self.prep.score_docs_by_zeroshot(df_dataset['text'], keyword)

        return score

    def score_by_topics(self, T, doc_ids, topic_weights):
        """
        Computes a score for every document in a given pandas dataframe
        according to the relevance of a weighted list of topics

        Parameters
        ----------
        T: numpy.ndarray or scipy.sparse
            Topic matrix (one column per topic)
        doc_ids: array-like
            Ids of the documents in the topic matrix. doc_ids[i] = '123' means
            that document with id '123' has topic vector T[i]
        topic_weights: dict
            Dictionary {t_i: w_i}, where t_i is a topic index and w_i is the
            weight of the topic

        Returns
        -------
        score : list of float
            List of scores, one per documents in corpus
        """

        # Create weight vector
        n_topics = T.shape[1]
        weights = np.zeros(n_topics,)
        # Convert key,values in dict into pos,value in array
        weights[list(topic_weights.keys())] = list(topic_weights.values())

        # Doc weights
        score = (T @ weights)

        # Add zero scores to docs not present in the topic matrix.
        # We do it by creating an auxiliary dataframe with all corpus ids
        df_temp = self.df_corpus[['id']].copy()
        df_temp['score'] = 0
        df_temp = df_temp.set_index('id')
        df_temp.loc[doc_ids, 'score'] = score
        score = df_temp['score']

        return score

    def get_top_scores(self, scores, n_max=1e100, s_min=0):
        """
        Select documents from the corpus whose score is strictly above a lower
        bound

        Parameters
        ----------
        scores: array-like of float
            List of scores. It must be the same size than the number of docs
            in the corpus
        n_max: int or None, optional (defaul=1e100)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=0)
            Minimum score. Only elements strictly above s_min are selected
        """

        ind = self.prep.get_top_scores(scores, n_max=n_max, s_min=s_min)
        ids = self.df_corpus.iloc[ind]['id']

        return ids

    def filter_by_keywords(self, keywords, wt=2, n_max=1e100, s_min=0,
                           model_name='all-MiniLM-L6-v2',
                           method='embedding'):
        """
        Select documents from a given set of keywords

        Parameters
        ----------
        keywords : list of str
            List of keywords
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor. Not used if
            self.path2embeddings is None
        n_max: int or None, optional (defaul=1e100)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=0)
            Minimum score. Only elements strictly above s_min are selected
        model_name : str, optinal (default = 'all-MiniLM-L6-v2')
            Name of the SBERT transformer model
        method : str in {'embedding', 'count'}
            - If 'count', documents are scored according to word counts
            - If 'embedding', scores are based on neural embeddings

        Returns
        -------
        ids : list
            List of ids of the selected documents
        scores : list of float
            List of scores, one per documents in corpus
        """

        scores = self.score_by_keywords(keywords, wt, model_name, method)
        ids = self.get_top_scores(scores, n_max=n_max, s_min=s_min)

        # ## return ids
        return ids, scores

    def filter_by_topics(self, T, doc_ids, topic_weights, n_max=1e100,
                         s_min=0):
        """
        Select documents with a significant presence of a given set of keywords

        Parameters
        ----------
        T: numpy.ndarray or scipy.sparse
            Topic matrix.
        doc_ids: array-like
            Ids of the documents in the topic matrix. doc_ids[i] = '123' means
            that document with id '123' has topic vector T[i]
        topic_weights: dict
            Dictionary {t_i: w_i}, where t_i is a topic index and w_i is the
            weight of the topic
        n_max: int or None, optional (defaul=1e100)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no loimit
        s_min: float, optional (default=0)
            Minimum score. Only elements strictly above s_min are selected

        Returns
        -------
        ids : list
            List of ids of the selected documents
        """

        score = self.score_by_topics(T, doc_ids, topic_weights)
        ids = self.get_top_scores(score, n_max=n_max, s_min=s_min)

        return ids

    def filter_by_zeroshot(self, keyword, n_max, s_min):

        scores = self.score_by_zeroshot(keyword)

        ids = self.get_top_scores(scores, n_max=n_max, s_min=s_min)

        return ids

    def make_pos_labels_df(self, ids):
        """
        Returns a dataframe with the given ids and a single, all-ones column

        Parameters
        ----------
        ids: array-like
            Values for the column 'ids'

        Returns
        -------
        df_labels: pandas.DataFrame
            A dataframe with two columns: 'id' and 'class'. All values in
            class column are equal to one.
        """

        df_labels = pd.DataFrame(columns=["id", "class"])
        df_labels['id'] = ids
        df_labels['class'] = 1

        return df_labels

    def make_PU_dataset(self, df_labels):
        """
        Returns the labeled dataframe in the format required by the
        CorpusClassifier class

        Parameters
        ----------
        df_corpus: pandas.DataFrame
            Text corpus, with at least three columns: id, title and description
        df_labels: pandas.DataFrame
            Dataframe of positive labels. It should contain column id. All
            labels are assumed to be positive

        Returns
        -------
        df_dataset: pandas.DataFrame
            A pandas dataframe with three columns: id, text and labels.
        """

        # Copy relevant columns only
        df_dataset = self.df_corpus.loc[:, ['id', 'title', 'description']]

        # Join title and description into a single column
        df_dataset.loc[:, 'text'] = (df_dataset['title'] + '. '
                                     + df_dataset['description'])
        df_dataset.drop(columns=['description', 'title'], inplace=True)

        # Default class is 0
        df_dataset['PUlabels'] = 0

        # Add positive labels
        df_dataset.loc[df_dataset.id.isin(df_labels.id), 'PUlabels'] = 1

        return df_dataset

    def evaluate_filter(
            self, scores, target_col, n_max, s_min, verbose=False):
        """
        Compute evaluation metrics for the generation of the subcorpus

        To do so, it requires from self.df_corpus to have at least the
        following columns: id, title, description, target_bio, target_tic,
        target_ene

        Parameters
        ----------
        scores : list of float
            list of unsorted scores
        target_col : str
            Name of the column in the corpus dataframe that will be used
            as a reference for evaluation.
        n_max: int or None, optional (defaul=1e100)
            Maximum number of elements in the output list. The default is
            a huge number that, in practice, means there is no limit
        s_min: float, optional (default=0)¡
            Minimum score. Only elements strictly above s_min are selected
        verbose : bool, optional
            If true, the evaluation results are logged at level INFO.

        Returns
        -------
        eval_scores : dict
            A dictionary of evaluation metrics. If there are no labels
            available for evaluation, an empty dictionary is returned.
        """

        target = self.df_corpus[target_col].to_numpy()
        eval_scores = self.prep.performance_metrics(
            scores, target, s_min, n_max)

        if verbose:
            logging.info(f'-- RESULTS for THRESHOLD {s_min}')
            logging.info(f"-- -- Accuracy = {eval_scores['acc_smin']:.4f}")
            logging.info(
                f"-- -- Balanced accuracy = {eval_scores['bal_acc_smin']:.4f}")
            logging.info(f"-- -- TPR (Recall) = {eval_scores['tpr_smin']:.4f}")
            logging.info(
                f"-- -- FPR (Fall-out) = {eval_scores['fpr_smin']:.4f}")
            # logging.info(f"-- -- Precision = {prfs[0][1]:.4f}")
            # logging.info(f"-- -- Recall = {prfs[1][1]:.4f}")
            # logging.info(f"-- -- F1 = {prfs[2][1]:.4f}")
            # logging.info(f"Average Precision = {ap:.4f}")

            logging.info(f'-- RESULTS for MAX_NUM DOCUMENTS {n_max}')
            logging.info(f"-- -- Accuracy = {eval_scores['acc_nmax']:.4f}")
            logging.info(
                f"-- -- Balanced accuracy = {eval_scores['bal_acc_nmax']:.4f}")
            logging.info(f"-- -- TPR (Recall) = {eval_scores['tpr_nmax']:.4f}")
            logging.info(
                f"-- -- FPR (Fall-out) = {eval_scores['fpr_nmax']:.4f}")

        return eval_scores
