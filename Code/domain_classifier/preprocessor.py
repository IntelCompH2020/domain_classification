class Preprocessor(object):
    """
    A container of corpus preprocessing methods
    """

    def __init__(self):
        """
        Initializes a preprocessor object
        """

        pass
        return

    def score_by_keywords(self, corpus, keywords):
        """
        Computes a score for every document in a given pandas dataframe
        according to the frequency of appearing some given keywords

        Parameters
        ----------
        corpus : dataframe
            Dataframe of corpus.

        keywords : list of str
            List of keywords

        Returns
        -------
        score : list of float
            List of scores, one per documents in corpus
        """

        score = []
        for doc in corpus:
            reps = [doc.count(k) for k in keywords]
            score.append(sum(reps))

        return score
