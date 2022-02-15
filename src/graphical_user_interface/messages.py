# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                           CLASS MESSAGES                              ***
******************************************************************************
Class containing the majority of the messages utilized in the GUI so as to
facilitate the readability of the code.
"""


class Messages:
    # Generic application messages
    ###########################################################################
    DC_MESSAGE = 'Domain classifier message'
    WINDOW_TITLE = 'Domain Classification'

    # Starting window messages
    ###########################################################################
    INCORRECT_INPUT_PARAM_SELECTION = 'A project folder and a source folder must be specified to proceed to the main ' \
                                      'menu. '
    # Toggle bar messages
    ###########################################################################
    INFO_LOAD_CORPUS_LABELS = 'Load corpus (thus resetting any previous corpus loaded in the project),\n ' \
                              'select a preliminary subcorpus from the positive class, and load or \n' \
                              'reset labels.'
    INFO_TRAIN_EVALUATE_PU = 'Train and evaluate a PU classifier model with the available labels.'
    INFO_GET_FEEDBACK = 'Get relevance feedback from user and update the classifier model with the latest relevance ' \
                        'feedback. '

    # Load corpus messages
    ###########################################################################
    INCORRECT_NO_CORPUS_SELECTED = 'You need to select a corpus first in order to proceed.'
    INFO_SELECT_CORPUS = 'Click on one of the corpus below listed and in the LOAD CORPUS button in order to load\n ' \
                         'an specific corpus into the application.'
    # Get labels messages
    ###########################################################################
    INCORRECT_NO_LABEL_OPTION_SELECTED = 'You must first choose how you want to load the subcorpus.'
    INFO_GET_LABELS = 'Select the option through which you want to load the labels and then clicked on the button\n ' \
                      'Get subcorpus. If you are a list of keywords to get the subcorpus, you must first select \n ' \
                      'the keywords that you want to use.'
    INFO_TYPE_KEYWORDS = 'Write your keywords (separated by commas, e.g.: "gradient descent, gibbs sampling") '
    NO_KEYWORDS_SELECTED = 'You must write a set of keywords in order to proceed.'
    INFO_TAG = 'Tag that is going to be used to compose the label file name.'
    NO_TAG_SELECTED = 'You must select a tag to compose the label file name'
    INFO_NO_ACTIVE_KEYWORDS = 'There are no active keywords in this session. Proceeding with its selection...'
    INFO_ACTIVE_KEYWORDS = 'Analyzing current list of active keywords....'
    INFO_TOPIC_LIST = 'In this line the weighted topic list that is going to be used for the attainment of the\n' \
                      'labels is going to be updated according to the weights that you are introducing on the top\n' \
                      'left table of this view with the following format: "id_0, weight_0, id_1, weight_1, ...".'
    NO_TOPIC_LIST_SELECTED = 'You must introduce your weighted topic list with the form:\n' \
                             '"id_0, weight_0, id_1, weight_1, ..."'
    INFO_TABLE_WEIGHTS = 'Insert the weight that you want to give for each of the associated topics located in the\n' \
                         'right table.'
    INFO_TABLE_PARAMETERS_KEYWORDS = 'These are the default values of the parameters needed for the labels import.\n' \
                                     'You can change any of them by tipping a different value on its corresponding\n' \
                                     'column. The parameters refer to:\n' \
                                     '- wt: Weighting factor for the title components.\n' \
                                     '- n_max: Maximum number of elements in the output list.\n' \
                                     '- s_min: Minimum score. Only elements strictly above s_min are selected.'

    INFO_TABLE_PARAMETERS_TOPICS = 'These are the default values of the parameters needed for the labels import.\n' \
                                   'You can change any of them by tipping a different value on its corresponding\n' \
                                   'column. The parameters refer to:\n' \
                                   '- n_max: Maximum number of elements in the output list.\n' \
                                   '- s_min: Minimum score. Only elements strictly above s_min are selected.'

    # Load labels messages
    ###########################################################################
    INFO_LOAD_RESET_LABELS = 'Select a set of labels and click on LOAD LABELS for loading it into the current \n' \
                             'session. Alternatively, click on RESET LABELS for removing previously selected labels.' \
 \
        # Train and evaluate PU model
    ###########################################################################
    WARNING_TRAINING = 'You must select a corpus and a set of labels in order to proceed with the training.'
    WARNING_EVALUATION = 'You must select a corpus and a set of labels in order to proceed with the evaluation.'
    INFO_TRAIN_PU_MODEL = 'The table at the top left shows the values of the parameters that are going to be used for ' \
                          'the training of the PU model; you can update them by inserting a new value and clicking ' \
                          'the corresponding button. To train the PU model (i.e. train a binary text classification ' \
                          'model based on transformers), you just need to click on the ' \
                          '"Train PU model" button; the training will start and until it is completed a loading bar ' \
                          'will be displayed. Once the training is completed, its logs are shown under the training ' \
                          'button. '
    INFO_EVALUATE_PU_MODEL = "By clicking on the 'Evaluate PU model' the predictions of the classification model over " \
                             "the input dataset and performance metrics are computed. "
    INFO_TABLE_TRAIN_PU_MODEL = 'With this table you can update the parameters of the classifier, namely:\n' \
                                '- max_imbalance: Maximum ratio negative vs positive samples in the training set' \
                                '- nmax: Maximum number of documents in the training set.' \
                                'The values shown are the default parameters; by inserting a new value and clicking ' \
                                '"Update parameters", you update the parameters; you can also reset them to its ' \
                                'default value by clicking on "Reset parameters." '

    # Getting relevance feedback messages
    ###########################################################################
    INFO_FEEDBACK = 'From this view, you can get your feedback about the trained model. To do so, as many documents ' \
                    'as specified in the parameter "n_docs", which you can configure at the right of this view, ' \
                    'will be displayed in the white spaces below, together with the prediction associated with each ' \
                    'of the documents, if it is available. Then, you must  write the labels that you want to use ' \
                    'for updating the model on the right column of the bottom table, and click on the button "Update ' \
                    'model based on feedback." '
    INFO_N_DOCS_AL = 'This parameter determines the number of documents to show each Active Learning round.\n By ' \
                     'default, it is configured to 5, but a different value can be assigned to it by inserting\n an ' \
                     'alternative number on the label located at the right. '
