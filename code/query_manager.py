import logging


class QueryManager(object):
    """
    This class contains all user queries needed by the datamanager.
    """

    def __init__(self):
        """
        Initializes the query manager object
        """

        pass

        return

    def ask_keywords(self, kw_library):
        """
        Ask the user for a list of keywords.

        Parameters
        ----------
        kw_library: list
            A list of possible keywords

        Returns
        -------
        keywords : list of str
            List of keywords
        """

        logging.info(
            f"-- Suggested list of keywords: {', '.join(kw_library)}\n")

        str_keys = input('-- Write your keywords (separated by commas, '
                         'e.g., "gradient descent, gibbs sampling") ')

        # Split by commas, removing leading and trailing spaces
        keywords = [x.strip() for x in str_keys.split(',')]
        # Remove multiple spaces
        keywords = [' '.join(x.split()) for x in keywords]

        return keywords

    def ask_label_tag(self):
        """
        Ask the user for a tag to compose the label file name.

        Returns
        -------
        keywords : list of str
            List of keywords
        """

        # Read available list of AI keywords
        tag = input('\n-- Write a tag for the new label file: ')

        return tag

    def ask_topics(self, topic_words):
        """
        Ask the user for a weighted list of topics

        Parameters
        ----------
        topic_words: list of str
            List of the main words from each topic

        Returns
        -------
        tw: dict
            Dictionary of topics: weights
        """

        logging.info("-- Topic descriptions: ")
        n_topics = len(topic_words)

        for i in range(n_topics):
            logging.info(f"-- -- Topic {i}: {topic_words[i]}")

        logging.info("")
        logging.info("-- Introduce your weigted topic list: ")
        logging.info("   id_0, weight_0, id_1, weight_1, ...")
        topic_weights_str = input(": ")

        tw_list = topic_weights_str.split(',')

        # Get topic indices as integers
        keys = [int(k) for k in tw_list[::2]]
        # Get topic weights as floats
        weights = [float(w) for w in tw_list[1::2]]

        # Normalize weights
        sum_w = sum(weights)
        weights = [w / sum_w for w in weights]

        # Store in dictionary
        tw = dict(zip(keys, weights))
        logging.info(f"-- Normalized weights: {tw}")

        return tw

