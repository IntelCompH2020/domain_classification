# -*- coding: utf-8 -*-
"""
Class containing the majority of the messages utilized in the GUI so as to facilitate the readability of the code.

@author: lcalv
"""


class Messages:
    DC_MESSAGE = 'Domain classifier message'
    WINDOW_TITLE = 'Domain Classification'
    INCORRECT_INPUT_PARAM_SELECTION = 'A project folder and a source folder must be specified to proceed to the main ' \
                                      'menu. '
    INCORRECT_NO_CORPUS_SELECTED = 'You need to select a corpus first in order to proceed with the label loading.'
    INCORRECT_NO_LABEL_OPTION_SELECTED = 'You must first choose how you want to load the subcorpus.'

    INFO_SELECT_CORPUS = 'Double click on one of the corpus below listed in order to load it into the application.'
    INFO_GET_LABELS = 'Select the option through which you want to load the labels and then clicked on the button\n ' \
                       'Get subcorpus. If you are a list of keywords to get the subcorpus, you must first select the\n '\
                       'keywords that you want to use.'
    INFO_LOAD_RESET_LABELS = ''
    INFO_TYPE_KEYWORDS = 'Write your keywords (separated by commas, e.g.: "gradient descent, gibbs sampling") '
    NO_KEYWORDS_SELECTED = 'You must write a set of keywords in order to proceed.'
    INFO_TAG = 'Write the tag to compose the label file name.'
    NO_TAG_SELECTED = 'You must select a tag to compose the label file name'
