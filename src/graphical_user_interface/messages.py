# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                           CLASS MESSAGES                              ***
******************************************************************************
Class containing the majority of the messages utilized in the GUI to
facilitate the readability of the code.
"""


class Messages:
    # Generic application messages
    ###########################################################################
    DC_MESSAGE = 'Domain classifier message'
    WINDOW_TITLE = 'Domain Classification'

    # Starting window messages
    ###########################################################################
    INCORRECT_INPUT_PARAM_SELECTION = \
        'A project folder and a source folder must be specified to proceed to the main menu.'

    # Load corpus messages
    ###########################################################################
    INCORRECT_NO_CORPUS_SELECTED = \
        'A corpus must be selected first in order to proceed.'
    INFO_SELECT_CORPUS = \
        "Select one of the corpus listed below and clicked the 'Load\n" \
        "corpus' button to load a specific corpus into the application."

    # Get labels messages
    ###########################################################################
    INCORRECT_NO_LABEL_OPTION_SELECTED = \
        "An option between:\n" \
        "1. Import labels from source file.\n" \
        "2. Get subcorpus from a given list of keywords.\n" \
        "3. Analyze the presence of selected keywords in corpus.\n" \
        "4. Get subcorpus from a topic selection function.\n" \
        "5. Get subcorpus from documents defining categories.\n" \
        "must be chosen before clicking on the 'Get labels' button to\n" \
        "determine the approach for selecting a preliminary subcorpus\n" \
        "from the positive class."
    NO_KEYWORDS_SELECTED = \
        'A set of keywords must be selected first in order to proceed.'
    NO_TAG_SELECTED = \
        "A tag to compose the label file name must be selected first."
    INFO_NO_ACTIVE_KEYWORDS = \
        'There are no active keywords in this session. Proceeding with its selection...'
    INFO_ACTIVE_KEYWORDS = \
        'Analyzing current list of active keywords....'
    NO_TOPIC_LIST_SELECTED = \
        "A weighted topic list with the form: 'id_0, weight_0, id_1," \
        "weight_1,...' must be selected to proceed to the next step."

    # Load labels messages
    ###########################################################################
    INFO_LOAD_RESET_LABELS = \
        "Select a set of labels and click the 'Load labels' button for\n" \
        "loading it into the current session. Alternatively, click the\n" \
        "'Reset labels' button to remove the previously selected labels."

    # Train and evaluate PU model
    ###########################################################################
    WARNING_TRAINING = \
        'To proceed with training: first select a corpus and set of labels.'
    WARNING_EVALUATION = \
        'To proceed with evaluation: first select a corpus and set of labels, and train a PU model'

    # Getting relevance feedback messages
    ###########################################################################

    # Train and revaluate PU model
    ###########################################################################
    WARNING_RETRAINING = \
        'To proceed with retraining: first select a corpus and set of labels.'
    WARNING_REEVALUATION = \
        'To proceed with reevaluation: first select a corpus and set of labels, and train a PU model'
