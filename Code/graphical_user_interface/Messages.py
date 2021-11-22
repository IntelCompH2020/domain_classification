# -*- coding: utf-8 -*-
"""
Class containing the majority of the messages utilized in the GUI so as to facilitate the readability of the code.

@author: lcalv
"""


class Messages:
    # Generic application messages
    ####################################################################################################################
    DC_MESSAGE = 'Domain classifier message'
    WINDOW_TITLE = 'Domain Classification'

    # Starting window messages
    ####################################################################################################################
    INCORRECT_INPUT_PARAM_SELECTION = 'A project folder and a source folder must be specified to proceed to the main ' \
                                      'menu. '
    # Toggle bar messages
    ####################################################################################################################
    INFO_LOAD_CORPUS_LABELS = 'Load corpus (thus resetting any previous corpus loaded in the project),\n ' \
                              'select a preliminary subcorpus from the positive class, and load or \n' \
                              'reset labels.'
    INFO_TRAIN_CLASSIFIER = 'Train classifier model with the available labels.'
    INFO_GET_FEEDBACK = 'Get relevance feedback to update the classifier model in accordance to it.'

    # Loading corpus messages
    ####################################################################################################################
    INCORRECT_NO_CORPUS_SELECTED = 'You need to select a corpus first in order to proceed.'
    INFO_SELECT_CORPUS = 'Click on one of the corpus below listed and in the LOAD CORPUS button in order to load\n ' \
                         'an specific corpus into the application.'

    # Getting labels messages
    ####################################################################################################################
    INCORRECT_NO_LABEL_OPTION_SELECTED = 'You must first choose how you want to load the subcorpus.'
    INFO_GET_LABELS = 'Select the option through which you want to load the labels and then clicked on the button\n ' \
                      'Get subcorpus. If you are a list of keywords to get the subcorpus, you must first select \n ' \
                      'the keywords that you want to use.'
    INFO_TYPE_KEYWORDS = 'Write your keywords (separated by commas, e.g.: "gradient descent, gibbs sampling") '
    NO_KEYWORDS_SELECTED = 'You must write a set of keywords in order to proceed.'
    INFO_TAG = 'Write the tag to compose the label file name.'
    NO_TAG_SELECTED = 'You must select a tag to compose the label file name'
    INFO_NO_ACTIVE_KEYWORDS = 'There are no active keywords in this session. Proceeding with its selection...'
    INFO_ACTIVE_KEYWORDS = 'Analyzing current list of active keywords...'
    INFO_TOPIC_LIST = 'Introduce your weighted topic list: "id_0, weight_0, id_1, weight_1, ..."'
    NO_TOPIC_LIST_SELECTED = 'You must introduce your weighted topic list with the form:\n'\
                             '"id_0, weight_0, id_1, weight_1, ..."'

    # Loading labels messages
    ####################################################################################################################
    INFO_LOAD_RESET_LABELS = 'Select a set of labels and click on LOAD LABELS for loading it into the current \n' \
                             'session. Alternatively, click on RESET LABELS for removing previously selected labels.' \

    # Training classifier messages
    ####################################################################################################################

    # Getting relevance feedback messages
    ####################################################################################################################





