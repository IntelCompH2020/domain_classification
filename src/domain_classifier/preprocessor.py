import numpy as np
import pandas as pd


class CorpusProcessor(object):
    """
    A container of corpus preprocessing methods

    It provides basic processing methods to a corpus of text documents

    The input corpus must be given by a list of strings (or a pandas series
    of strings)
    """

    def __init__(self):
        """
        Initializes a preprocessor object
        """

        return

    def score_docs_by_keywords(self, corpus, keywords):
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


class CorpusDFProcessor(object):
    """
    A container of corpus processing methods.
    It assumes that a corpus is given by a dataframe of documents.
    Each dataframe must contain three columns:
        id: document identifiers
        title: document titles
        description: body of the document text
    """

    def __init__(self, df_corpus):
        """
        Initializes a preprocessor object
        """

        self.df_corpus = df_corpus

        # This class uses methods from the corpus processor class.
        # FIXME: Uset CorpusProcessor as a base class and inherit
        #        CorpusDFProcessor from CorpusProcessor
        self.prep = CorpusProcessor()

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

    def score_by_keywords(self, keywords, wt=2):
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

        score_title = self.prep.score_docs_by_keywords(
            self.df_corpus['title'], keywords)
        score_descr = self.prep.score_docs_by_keywords(
            self.df_corpus['description'], keywords)

        score = wt * np.array(score_title) + np.array(score_descr)

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

    def filter_by_keywords(self, keywords, wt=2, n_max=1e100, s_min=0):
        """
        Select documnts with a significant presence of a given set of keywords

        Parameters
        ----------
        keywords : list of str
            List of keywords
        wt : float, optional (default=2)
            Weighting factor for the title components. Keyword matches with
            title words are weighted by this factor
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

        scores = self.score_by_keywords(keywords, wt)
        ids = self.get_top_scores(scores, n_max=n_max, s_min=s_min)

        return ids

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
        df_dataset = self.df_corpus[['id', 'title', 'description']]

        # Joing title and description into a single column
        df_dataset['text'] = (df_dataset['title'] + '.'
                              + df_dataset['description'])
        df_dataset.drop(columns=['description', 'title'], inplace=True)

        # Default class is 0
        df_dataset['labels'] = 0

        # Add positive labels
        df_dataset.loc[df_dataset.id.isin(df_labels.id), 'labels'] = 1

        return df_dataset
