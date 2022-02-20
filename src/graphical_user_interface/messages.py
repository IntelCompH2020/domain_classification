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
    INCORRECT_INPUT_PARAM_SELECTION = \
        'A project folder and a source folder must be specified to proceed to the main menu.'
    # Toggle bar messages
    ###########################################################################
    INFO_LOAD_CORPUS_LABELS = \
        'Load corpus, select a preliminary subcorpus from the positive class,\n' \
        'and load or reset labels.'
    INFO_TRAIN_EVALUATE_PU = \
        'Train and evaluate a PU classifier model with the available labels.'
    INFO_GET_FEEDBACK = \
        'Get relevance feedback from user and update the classifier\n' \
        'model with the latest relevance feedback. '

    # Load corpus messages
    ###########################################################################
    INCORRECT_NO_CORPUS_SELECTED =\
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
        "must be chosen before clicking on the 'Get labels' button in order to\n" \
        "determine the approach for selecting a preliminary subcorpus from the\n" \
        "positive class."
    INFO_GET_LABELS = \
        "elect the option through which you want to select a preliminary subcorpus\n" \
        "from the positive class and then click the button 'Get subcorpus'."
    INFO_TYPE_KEYWORDS =\
        'Write your keywords (separated by commas, e.g.:\n"gradient descent, gibbs sampling, etc.") '
    NO_KEYWORDS_SELECTED = \
        'A set of keywords must be selected first in order to proceed.'
    INFO_TAG = \
        'Tag to be used to compose the label file name.'
    NO_TAG_SELECTED = \
        "A tag to compose the label file name must be selected first."
    INFO_NO_ACTIVE_KEYWORDS = \
        'There are no active keywords in this session. Proceeding with its selection...'
    INFO_ACTIVE_KEYWORDS = \
        'Analyzing current list of active keywords....'
    INFO_TOPIC_LIST =\
        'In this line the weighted topic list to be used for the labels attainment\n' \
        'will be updated according to the weights that you are introducing on the top\n' \
        'left table of this view with the following format: "id_0, weight_0, id_1, weight_1, ...".'
    NO_TOPIC_LIST_SELECTED =\
        "A weighted topic list with the form: 'id_0, weight_0, id_1, weight_1, ...'\n"\
        "must be selected before proceeding to the next step."
    INFO_TABLE_WEIGHTS = 'Insert a weight for each of the associated topics located in the right table.'
    INFO_TABLE_PARAMETERS_KEYWORDS =\
        'These are the default values of the parameters needed for the labels import.\n' \
        'Any of them can be changed by tipping a different value on its corresponding\n' \
        'column. The parameters refer to:\n' \
        '- wt: Weighting factor for the title components.\n' \
        '- n_max: Maximum number of elements in the output list.\n' \
        '- s_min: Minimum score. Only elements strictly above s_min are selected.'
    INFO_TABLE_PARAMETERS_TOPICS =\
        'These are the default values of the parameters needed for the labels import.\n' \
        'Any of them can be changed by tipping a different value on its corresponding\n' \
        'column. The parameters refer to:\n' \
        '- n_max: Maximum number of elements in the output list.\n' \
        '- s_min: Minimum score. Only elements strictly above s_min are selected.'

    # Load labels messages
    ###########################################################################
    INFO_LOAD_RESET_LABELS = \
        "Select a set of labels and click the 'Load labels' button for loading it into\n" \
        "the current session. Alternatively, click the button 'Reset labels' for removing\n" \
        "the previously selected labels."

    # Train and evaluate PU model
    ###########################################################################
    WARNING_TRAINING = \
        'A corpus and a set of labels must be selected first in order to proceed with the training.'
    WARNING_EVALUATION = \
        'A corpus and a set of labels must be selected first in order to proceed with the evaluation.'
    INFO_TRAIN_PU_MODEL = \
        'The table at the top left shows the values of the parameters that are used\n' \
        'for the training of the PU model; you can update them by inserting a new\n' \
        'value and clicking the corresponding button. To train the PU model (i.e.\n' \
        'train a binary text classification model based on transformers), you just\n' \
        'need to click the "Train PU model" button; the training will start and until\n' \
        'it is completed a loading bar and its corresponding logs will be displayed.'
    INFO_EVALUATE_PU_MODEL = \
        "By clicking on the 'Evaluate PU model' button the predictions of the\n" \
        "classification model over the input dataset and performance metrics are\n" \
        "computed. Once completed, the results are shown under the 'CLASSIFICATION\n" \
        "RESULTS' label (at the bottom right table)."
    INFO_TABLE_TRAIN_PU_MODEL = \
        'With this table you can update the parameters of the classifier, namely:\n' \
        '- max_imbalance: Maximum ratio negative vs positive samples in the training set.\n' \
        '- nmax: Maximum number of documents in the training set.\n' \
        'The values shown are the default parameters; by inserting a new value and clicking\n' \
        '"Update parameters", the parameters are updated; they can also be restored to its\n' \
        'default value by clicking on "Reset parameters." '

    # Getting relevance feedback messages
    ###########################################################################
    INFO_FEEDBACK =\
        'From this view, you can get your feedback about the trained model.\n' \
        'To do so, as many documents as specified in the parameter "n_docs",\n' \
        'which you can configure at the right of this view, will be displayed\n' \
        'in the white spaces below, together with the prediction associated with\n' \
        'each of the documents, if it is available. Documents that initially belong\n' \
        'to the positive class (1) are highlighted in purple, those that belong to\n' \
        'the negative class (0) in red. To annotate a document as belonging to the\n' \
        'the positive class, the checkbox that is located under such a document must\n' \
        'be checked. Once the annotation of the documents has been completed, the\n' \
        'button "Give feedback" must be clicked. Once this is done, the model can\n' \
        'be retrained using the just inserted manual annotations by clicking the\n' \
        '"Retrain model" button, and reevaluated by clicking on "Reevaluate model". '
    INFO_N_DOCS_AL =\
        'This parameter determines the number of documents to show each Active Learning round.\n By ' \
        'default, it is configured to 4, but a different value can be assigned to it by inserting\n an ' \
        'alternative number on the label located at the right. '
