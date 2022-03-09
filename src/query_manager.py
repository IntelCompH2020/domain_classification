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

    def ask_keywords(self, kw_library=None):
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

        if kw_library is not None:
            logging.info(
                f"-- Suggested list of keywords: {', '.join(kw_library)}\n")

        str_keys = input('-- Write your keyword/s (separated by commas, '
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

    def ask_value(self, query="Write value: ", convert_to=str, default=None):
        """
        Ask user for a value

        Parameters
        ----------
        query: str, optional (default value="Write value: ")
            Text to print
        convert_to: function, optional (default=str)
            A function to apply to the selected value. It can be used, for
            instance, for type conversion.
        default: optional (default=None)
            Default value to return if an empty value is returned

        Returns
        -------
        value:
            The returned value is equal to conver_to(x), where x is the string
            introduced by the user (if any) or the default value.
        """

        if default:
            x = input(f"{query} [default: {default}]: ")
        else:
            x = input(query)

        if x:
            # Convert to
            value = convert_to(x)
        else:
            value = default

        return value

    def ask_label(self):
        """
        Ask the user for a single binary label

        Returns
        -------
        label: int
            Binary value read from the standard input
        """

        logging.info("")
        logging.info(
            "-- Class of the document (1 for positive class, 0 for negative)")
        label = input(": ")

        while label not in {'0', '1'}:
            logging.info("-- Try it again: the label must be 0 or 1")
            label = input(": ")

        # Split and convert to integers
        label = int(label)

        return label

    def ask_labels(self):
        """
        Ask the user for a weighted list of labels related to some documents

        Returns
        -------
        labels: list of int
            List of labels
        """

        logging.info("")
        logging.info("-- Provide a sequence of labels for the given documents "
                     " in the same order: ")
        labels_str = input(": ")

        # Split and convert to integers
        labels = [int(k) for k in labels_str.split(',')]

        return labels

    def confirm(self):
        """
        Ask the user for confirmation

        Returns
        -------
        True if the user inputs 'y', False otherwise
        """

        logging.info("")
        label = input("-- Press 'y' to confirm the anotation. Otherwise all "
                      "labels in this session will be removed: ")

        return label == 'y'

