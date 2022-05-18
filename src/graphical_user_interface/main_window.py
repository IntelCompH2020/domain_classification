# -*- coding: utf-8 -*-

"""
@author: L. Calvo-Bartolome
"""

import numpy as np
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtWidgets import QButtonGroup, QDesktopWidget, QTextEdit, QCheckBox
from PyQt5.QtCore import QThreadPool, QSize, pyqtSlot
from PyQt5.QtGui import QPixmap
from functools import partial

# Local imports
from src.graphical_user_interface.analyze_keywords_window import (
    AnalyzeKeywordsWindow)
from src.graphical_user_interface.get_category_names_window import GetCategoryNamesWindow
from src.graphical_user_interface.get_keywords_window import GetKeywordsWindow
from src.graphical_user_interface.get_topics_list_window import (
    GetTopicsListWindow)
from src.graphical_user_interface.messages import Messages
from src.graphical_user_interface.util import toggle_menu, execute_in_thread, change_background_color_text_edit
from src.graphical_user_interface.constants import Constants


class MainWindow(QtWidgets.QMainWindow):
    """
    Class representing the main window of the application.
    """

    def __init__(self, project_folder, source_folder, tm, widget, stdout, stderr):
        """
        Initializes the application's main window based on the parameters received
        from the application's starting window.

        Parameters
        ----------
        project_folder : pathlib.Path
            Path to the application project
        source_folder : pathlib.Path
            Path to the folder containing the data sources
        tm : TaskManager 
            TaskManager object associated with the project
        widget : QtWidgets.QStackedWidget
            Window to which the application's main window is attached to
        stdout : sys.stdout
            Output file object
        stderr : sys.stderr
            Standard Error file object
        """

        super(MainWindow, self).__init__()

        # Load UI and configure default geometry of the window
        #######################################################################
        uic.loadUi("UIS/DomainClassifier.ui", self)
        self.init_ui()
        self.animation = QtCore.QPropertyAnimation(self.frame_left_menu,
                                                   b"minimumWidth")

        # ATTRIBUTES
        #######################################################################
        # Attributes coming from the starting window
        self.source_folder = source_folder
        self.project_folder = project_folder
        self.tm = tm
        self.widget = widget

        # Attributes for the maintenance of the options currently selected
        self.corpus_selected_name = ""
        self.labels_loaded = None
        self.get_label_option = 0

        # Attribute for saving the messages that come from the TM that are going to be displayed with pop-up windows
        self.message_out = None

        # Creation of subwindows for the selection of labels by keywords, topics and the analysis of keywords
        self.get_keywords_window = GetKeywordsWindow(tm)
        self.analyze_keywords_window = AnalyzeKeywordsWindow(tm)
        self.get_topics_list_window = GetTopicsListWindow(tm)
        self.get_category_names_window = GetCategoryNamesWindow(tm)

        # Attributes that define the parameters read from the configuration file. Initially, the current values are
        # initialized to be equal to the default values read from the configuration file ("parameters.default.yml")
        self.class_max_imbalance_dft = self.tm.global_parameters['classifier']['max_imbalance']
        self.class_max_imbalance = self.class_max_imbalance_dft
        self.class_nmax_dft = self.tm.global_parameters['classifier']['nmax']
        self.class_nmax = self.class_nmax_dft
        self.n_docs_al_dft = self.tm.global_parameters['active_learning']['n_docs']
        self.n_docs_al = self.n_docs_al_dft

        # Attributes for saving the dictionary of results associated with a PU model's evaluation and reevalution
        self.result_evaluation_pu_model = None
        self.result_reevaluation_pu_model = None

        # Attributes for keeping the state of the feedback's annotations from one method to the other
        self.selected_docs_to_annotate = None
        self.idx_docs_to_annotate = None
        self.labels_docs_to_annotate_dict = {}
        self.labels_docs_to_annotate = []

        # Attributes to redirect stdout and stderr
        self.stdout = stdout
        self.stderr = stderr

        # #####################################################################
        # THREADS FOR EXECUTING IN PARALLEL
        # #####################################################################
        self.thread_pool = QThreadPool()
        print("Multithreading with maximum"
              " %d threads" % self.thread_pool.maxThreadCount())

        # #####################################################################
        # CONFIGURE ELEMENTS IN THE "LOAD CORPUS VIEW"
        # #####################################################################
        # The loading bar is initially not visible
        self.progress_bar_first_tab.setVisible(False)
        self.progress_bar_first_tab.setValue(0)

        # LOAD CORPUS WIDGETS
        # #####################################################################
        self.load_corpus_push_button.clicked.connect(self.clicked_load_corpus)

        # The corpora available withing the source_folder are direclty displayed
        self.show_corpora()

        # GET LABELS WIDGETS
        # #####################################################################
        self.get_labels_radio_buttons = QButtonGroup(self)
        self.get_labels_radio_buttons.addButton(self.get_labels_option_1, 1)
        self.get_labels_radio_buttons.addButton(self.get_labels_option_2, 2)
        self.get_labels_radio_buttons.addButton(self.get_labels_option_3, 3)
        self.get_labels_radio_buttons.addButton(self.get_labels_option_4, 4)
        self.get_labels_radio_buttons.addButton(self.get_labels_option_5, 5)
        self.get_labels_radio_buttons.buttonClicked.connect(
            self.clicked_get_labels_option)
        self.get_labels_push_button.clicked.connect(self.clicked_get_labels)

        # LOAD LABELS WIDGETS
        # #####################################################################
        self.load_labels_push_button.clicked.connect(self.clicked_load_labels)
        self.reset_labels_push_button.clicked.connect(
            self.clicked_reset_labels)

        # TRAIN AND EVALUATE PU MODEL WIDGETS
        # #####################################################################
        self.train_pu_model_push_button.clicked.connect(
            self.clicked_train_PU_model)
        self.eval_pu_classifier_push_button.clicked.connect(
            self.clicked_evaluate_PU_model)
        self.update_param_pu_model_push_button.clicked.connect(
            self.update_params_train_pu_model)
        self.reset_param_training_pu_model_push_button.clicked.connect(
            self.reset_params_train_pu_model)

        # The loading bars are initially not visible
        self.progress_bar_train.setVisible(False)
        self.progress_bar_train.setValue(0)
        self.progress_bar_evaluate.setVisible(False)
        self.progress_bar_evaluate.setValue(0)

        # The PU model training's parameters are initialized at its corresponding table
        # from the values read from the configuration file
        self.init_params_train_pu_model()

        self.table_pu_classification_results.resizeColumnsToContents()
        self.table_pu_classification_results.resizeRowsToContents()
        self.table_pu_classification_results.verticalHeader().setVisible(True)

        self.table_train_pu_model_params.horizontalHeader().setVisible(True)

        # GET FEEDBACK WIDGETS
        # #####################################################################
        self.give_feedback_push_button.clicked.connect(
            self.clicked_give_feedback)
        self.update_ndocs_al_push_button.clicked.connect(self.clicked_update_ndocs_al)
        self.retrain_model_push_button.clicked.connect(
            self.clicked_retrain_model)
        self.reevaluate_model_push_button.clicked.connect(
            self.clicked_reevaluate_model)

        checkboxes_predictions = []
        for id_checkbox in np.arange(Constants.MAX_N_DOCS):
            doc_checkbox_name = "prediction_doc_" + str(id_checkbox + 1)
            doc_checkbox_widget = self.findChild(QCheckBox, doc_checkbox_name)
            checkboxes_predictions.append(doc_checkbox_widget)

            # Initialize all predictions as belonging to the negative class
            self.labels_docs_to_annotate_dict[doc_checkbox_name] = 0

        for checkbox_pred in checkboxes_predictions:
            checkbox_pred.stateChanged.connect(partial(self.clicked_change_predicted_class, checkbox_pred))

        # The loading bar is initially not visible
        self.progress_bar_feedback_update.setVisible(False)
        self.progress_bar_feedback_update.setValue(0)

        self.table_reclassification_results.resizeColumnsToContents()
        self.table_reclassification_results.resizeRowsToContents()
        self.table_reclassification_results.verticalHeader().setVisible(True)

        # Initialize the value for the number of documents to annotate based on the default value noted in the
        # configuration file
        self.init_ndocs_al()

        # ####################################################################
        # TOGGLE MENU
        # #####################################################################
        self.toggleButton.clicked.connect(lambda: toggle_menu(self, 250))

        # PAGES
        # #####################################################################
        # PAGE 1: Load corpus/ labels
        self.pushButtonLoad.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_load))

        # PAGE 2: Train classifier
        self.pushButtonTrain.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_train))

        # PAGE 3: Get relevance feedback
        self.pushButtonGetFeedback.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_feedback))

    def init_ui(self):
        """
        Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        self.setWindowIcon(QIcon('UIs/Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.resize(self.minimumSizeHint())
        self.center()

        return

    def center(self):
        """
        Centers the window at the middle of the screen at which the application is being executed.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        return

    # #################################################################################################################
    # LOAD CORPUS FUNCTIONS
    # #################################################################################################################
    def show_corpora(self):
        """
        List all corpora contained in the source folder selected by the user.
        """
        corpus_list = self.tm.DM.get_corpus_list()
        for corpus_nr in np.arange(0, len(corpus_list), 1):
            self.tree_view_select_corpus.insertItem(
                corpus_nr, corpus_list[corpus_nr])

        # If a corpus was already loaded in the current project, such a corpus is shown as selected corpus
        if self.tm.state['selected_corpus']:
            current_corpus = self.tm.metadata['corpus_name']

            # Show message informing about the selection of the corpus
            informative = "The corpus of this project is " + current_corpus + "."
            QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE, informative)

            # Write the name of the selected corpus (i.e. current project's associated corpus)
            self.label_corpus_selected_is.setText(str(current_corpus))
            self.show_labels()

        return

    def execute_load_corpus(self):
        """
        Method to control the execution of the loading of a corpus on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """
        self.tm.load_corpus(self.corpus_selected_name)

        return "Done."

    def do_after_load_corpus(self):
        """
        Method to be executed after the loading of the corpus has been
        completed.
        """
        # Hide progress bar
        self.progress_bar_first_tab.setVisible(False)

        # Show messages in the status bar, pop up window, and corpus label
        self.statusBar().showMessage(
            "'" + self.corpus_selected_name + "' was selected as corpus.",
            Constants.LONG_TIME_SHOW_SB)
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The corpus '" + self.corpus_selected_name + "' has been "
                                                         "loaded in the current session.")

        # Write the name of the selected corpus
        self.label_corpus_selected_is.setText(str(self.corpus_selected_name))

        # Show associated labels
        self.show_labels()

        return

    def clicked_load_corpus(self):
        """
        Method to control the selection of a new corpus by double-clicking
        one of the items of the corpus list within the selected source folder,
        as well as its loading as dataframe into the TaskManager object.
        Important is that the corpus cannot be changed inside the same project,
        so if a corpus was used before me must keep the same one.
        """

        # Get item of the TreeWidget that the user has selected
        item = self.tree_view_select_corpus.currentItem()
        if item is not None:
            corpus_name = str(item.text())
            current_corpus = self.tm.metadata['corpus_name']

            # Go back to the main window of the application so the user can select
            # a different project folder in case he chooses a different corpus for
            # an already existing project
            if self.tm.state['selected_corpus'] and corpus_name != current_corpus:
                # Show warning message
                warning = "The corpus of this project is " + current_corpus + \
                          ". Run another project to use " + corpus_name + "."
                QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, warning)

                # Return to the starting window
                self.widget.removeWidget(self.widget.currentWidget())
                return

            # Save selected corpus as attribute to be accessible from other methods
            self.corpus_selected_name = corpus_name

            # Load corpus into the TaskManager object as dataframe
            execute_in_thread(
                self, self.execute_load_corpus, self.do_after_load_corpus, self.progress_bar_first_tab)

        return

    # #########################################################################
    # GET LABELS FUNCTIONS
    # #########################################################################
    def execute_import_labels(self):
        """
        Imports the labels by invoking the corresponding method in the Task Manager
        object associated with the GUI.
        """
        # Get labels
        if self.get_label_option == 1:
            message_out = self.tm.import_labels()
        elif self.get_label_option == 2 or self.get_label_option == 3:
            message_out = self.tm.get_labels_by_keywords(
                keywords=self.get_keywords_window.selectedKeywords,
                wt=self.get_keywords_window.wt,
                n_max=self.get_keywords_window.n_max,
                s_min=self.get_keywords_window.s_min,
                tag=self.get_keywords_window.selectedTag)
        elif self.get_label_option == 4:
            message_out = self.tm.get_labels_by_topics(
                topic_weights=self.get_topics_list_window.tw,
                T=self.get_topics_list_window.T,
                df_metadata=self.get_topics_list_window.df_metadata,
                n_max=self.get_topics_list_window.n_max,
                s_min=self.get_topics_list_window.s_min,
                tag=self.get_topics_list_window.selectedTag)
        elif self.get_label_option == 5:
            message_out = self.tm.get_labels_by_zeroshot(
                keywords=self.get_category_names_window.selectedKeywords,
                n_max=self.get_category_names_window.n_max,
                s_min=self.get_category_names_window.s_min,
                tag=self.get_category_names_window.selectedTag)

        self.message_out = message_out[3:]

        return "Done"

    def do_after_import_labels(self):
        """
        Function to be executed after the labels' importing has been completed.
        """
        # Hide progress bar
        self.progress_bar_first_tab.setVisible(False)

        # Show informative message
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE, self.message_out)

        if self.get_label_option == 3:
            # Show the window for the analysis of the keywords
            self.analyze_keywords_window.do_analysis()
            self.analyze_keywords_window.exec()

        # Reset after loading labels
        self.get_labels_radio_buttons.setExclusive(False)
        self.get_labels_option_1.setChecked(False)
        self.get_labels_option_2.setChecked(False)
        self.get_labels_option_3.setChecked(False)
        self.get_labels_option_4.setChecked(False)
        self.get_labels_option_5.setChecked(False)
        self.get_labels_radio_buttons.setExclusive(True)
        self.get_label_option == 0

        # Load just gotten labels
        self.show_labels()

        return

    def clicked_get_labels_option(self):
        """
        Method to control the functionality associated with the selection of
        each of the QRadioButtons associated with the labels' getting.
        Only one QRadioButton can be selected at a time.
        """
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_INPUT_PARAM_SELECTION)
        else:
            if self.get_labels_radio_buttons.checkedId() == 1:
                print("Import labels from a source file")
                self.get_label_option = 1
            elif self.get_labels_radio_buttons.checkedId() == 2:
                print("Get subcorpus from a given list of keywords")
                self.get_label_option = 2
                # Show the window for selecting the keywords
                self.get_keywords_window.show_suggested_keywords()
                self.get_keywords_window.exec()
            elif self.get_labels_radio_buttons.checkedId() == 3:
                print("Analyze the presence of selected keywords in the "
                      "corpus")
                # Show the window for selecting the keywords in case they
                # have not been selected yet
                if self.tm.keywords is None:
                    self.get_label_option = 3
                    QtWidgets.QMessageBox.information(
                        self, Messages.DC_MESSAGE,
                        Messages.INFO_NO_ACTIVE_KEYWORDS)
                    # Show the window for selecting the keywords
                    self.get_keywords_window.show_suggested_keywords()
                    self.get_keywords_window.exec()
                else:
                    self.get_label_option = - 1
                    QtWidgets.QMessageBox.information(
                        self, Messages.DC_MESSAGE,
                        Messages.INFO_ACTIVE_KEYWORDS)
                    # Show the window for the analysis of the keywords
                    self.analyze_keywords_window.do_analysis()
                    self.analyze_keywords_window.exec()

            elif self.get_labels_radio_buttons.checkedId() == 4:
                print("Get subcorpus from a topic selection function")
                self.get_label_option = 4
                # Show the window for selecting the topics
                self.get_topics_list_window.show_topics()
                self.get_topics_list_window.exec()
            else:
                print("Get subcorpus from documents defining categories")
                self.get_label_option = 5
                # Show the window for selecting the categories of the zero-shot model
                self.get_category_names_window.exec()

        return

    def clicked_get_labels(self):
        """
        Method for performing the getting of the labels according to the
        method selected for it.
        """
        if self.get_label_option == 0:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_LABEL_OPTION_SELECTED)
            return

        execute_in_thread(self, self.execute_import_labels,
                          self.do_after_import_labels, self.progress_bar_first_tab)

        return

    # #########################################################################
    # LOAD LABELS FUNCTIONS
    # #########################################################################
    def show_labels(self):
        """
        Method for showing the labels associated with the selected corpus.
        """

        # Clear previous content from the QTreeWidget
        self.tree_view_load_labels.clear()

        # Show warning message indicating that a corpus needs to be selected first
        # in order to proceed with the labels' loading
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            # Get and inserts elements in the corresponding QTreeWidget
            labelset_list = self.tm.DM.get_labelset_list()
            for labelset_nr in np.arange(0, len(labelset_list), 1):
                self.tree_view_load_labels.insertItem(
                    labelset_nr, labelset_list[labelset_nr])

        return

    def clicked_load_labels(self):
        """
        Method for controlling the loading of the labels into the session.
        It is equivalent to the "_get_labelset_list" method from the
        TaskManager class
        """

        # Show warning message if not corpus has been selected before asking for the labels' loading
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            # Load labels by invoking the corresponding TM's function
            item = self.tree_view_load_labels.currentItem()
            self.labels_loaded = str(item.text())
            self.tm.load_labels(self.labels_loaded)

            # Show messages in the status bar, pop up window, and corpus
            # label
            self.statusBar().showMessage(
                "'" + self.labels_loaded + "' were loaded.", Constants.LONG_TIME_SHOW_SB)
            QtWidgets.QMessageBox.information(
                self, Messages.DC_MESSAGE,
                "The labels '" + self.labels_loaded + "' have been loaded"
                                                      " in the current session.")

            # Write the name of the laoded set of labels in the "label_labels_loaded_are" QLineEdit
            self.label_labels_loaded_are.setText(str(self.labels_loaded))

            # Show documents at the GET FEEDBACK if the conditions for it are satisfied
            self.init_feedback_elements()

        return

    def clicked_reset_labels(self):
        """
        Method for controlling the resetting of the current session's labels.
        """

        # Show warning message if not corpus has been selected before asking for the labels' loading
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            # Reset labels by invoking the corresponding TM's function
            item = self.tree_view_load_labels.currentItem()
            self.labels_loaded = str(item.text())
            if self.labels_loaded is not None:
                self.tm.reset_labels(self.labels_loaded)
            aux_labels = self.labels_loaded
            self.labels_loaded = None

            # Show messages in the status bar, pop up window, and corpus
            # label
            self.statusBar().showMessage(
                "'" + aux_labels + "' were removed.", Constants.LONG_TIME_SHOW_SB)
            QtWidgets.QMessageBox.information(
                self, Messages.DC_MESSAGE,
                "The labels '" + aux_labels + "' have been removed from the "
                                              "current session.")

            # Remove the previously loaded labels from "label_labels_loaded_are"
            self.label_labels_loaded_are.setText(str(" "))

            # Reload labels
            self.show_labels()

        return

    # #########################################################################
    # TRAIN PU MODEL FUNCTIONS
    # #########################################################################
    @pyqtSlot(str)
    def append_text_train(self, text):
        """
        Method to redirect the stdout and stderr in the "text_logs_train_pu_model"
        while the training of a PU model is being performed.
        """

        self.text_logs_train_pu_model.moveCursor(QTextCursor.End)
        self.text_logs_train_pu_model.insertPlainText(text)

        return

    def init_params_train_pu_model(self):
        """
        Initializes the classifier parameters in the parameters' table within the second tab of the main GUI
        window, i.e. max_imbalance and nmax. The default configuration of these parameters is read from the
        configuration file '/config/parameters.default.yaml'.
        """

        # Clear previous contents and set number of rows and columns
        self.table_train_pu_model_params.clearContents()
        self.table_train_pu_model_params.setRowCount(1)
        self.table_train_pu_model_params.setColumnCount(2)

        # Fill the table
        self.table_train_pu_model_params.setItem(
            0, 0, QtWidgets.QTableWidgetItem(str(self.class_max_imbalance)))
        self.table_train_pu_model_params.setItem(
            0, 1, QtWidgets.QTableWidgetItem(str(self.class_nmax)))

        # Set horizontal header visible
        self.table_train_pu_model_params.horizontalHeader().setVisible(True)

        return

    def update_params_train_pu_model(self):
        """
        Updates the classifier parameters that are going to be used for the training of the PU model based on the
        values read from the table within the second tab of the main GUI
        window that have been specified by the user.
        """

        # Get new parameters written in the table by the user
        if self.table_train_pu_model_params.item(0, 0) is not None:
            self.class_max_imbalance = int(self.table_train_pu_model_params.item(0, 0).text())
        else:
            self.class_max_imbalance = self.class_max_imbalance_dft

        if self.table_train_pu_model_params.item(0, 1) is not None:
            self.class_nmax = int(self.table_train_pu_model_params.item(0, 1).text())
        else:
            self.class_nmax = self.class_nmax_dft

        # Show informative message with the changes
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The parameters of classifier has been set to '" + str(self.class_max_imbalance) + "' and '" + \
            str(self.class_nmax) + "' for the current session.")

        return

    def reset_params_train_pu_model(self):
        """
        Resets the PU model training parameters to its default value based on the values
        that were read initially from the configuration file
        """

        # Get default values
        self.class_max_imbalance = self.class_max_imbalance_dft
        self.class_nmax = self.class_nmax_dft

        # Rewrite the values in the corresponding training parameters' table
        self.init_params_train_pu_model()

        # Show informative message with the changes
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The parameters of classifier has been set to its default value.")

        return

    def execute_train_classifier(self):
        """
        Method to control the execution of the training of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """

        # Connect pyslots for the stdout and stderr redirection during the time
        # the training is being performed
        self.stdout.outputWritten.connect(self.append_text_train)
        self.stderr.outputWritten.connect(self.append_text_train)

        # Train the PU model by invoking the task manager method
        self.tm.train_PUmodel(self.class_max_imbalance, self.class_nmax)

        return "Done."

    def do_after_train_classifier(self):
        """
        Method to be executed after the training of the classifier has been
        completed.
        """

        # Hide progress bar
        self.progress_bar_train.setVisible(False)

        # Disconnect from stdout and stderr
        self.stdout.outputWritten.disconnect()
        self.stderr.outputWritten.disconnect()

        # Show message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The PU model has been trained.")

        return

    def clicked_train_PU_model(self):
        """
        Method that controls the actions that are carried out when the button "train_pu_model_push_button" is
        clicked by the user.
        """

        # Check if a corpus has been selected. Otherwise, the training cannot be carried out
        if self.corpus_selected_name is not None and self.tm.df_labels is not None:

            # Execute the PU model training in the secondary thread
            execute_in_thread(
                self, self.execute_train_classifier, self.do_after_train_classifier,
                self.progress_bar_train)
        else:

            # Show warning message if training could not be carried out as either no corpus or
            # not labels were selected at the time the request was made
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.WARNING_TRAINING)

        return

    # #########################################################################
    # EVALUATE PU MODEL FUNCTIONS
    # #########################################################################
    @pyqtSlot(str)
    def append_text_evaluate(self, text):
        """
        Method to redirect the stdout and stderr in the "text_edit_results_eval_pu_classifier"
        while the evaluation of a PU model is being performed.
        """

        self.text_edit_results_eval_pu_classifier.moveCursor(QTextCursor.End)
        self.text_edit_results_eval_pu_classifier.insertPlainText(text)

    def execute_evaluate_pu_model(self):
        """
        Method to control the execution of the evaluation of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """

        # Connect pyslots for the stdout and stderr redirection during the time
        # the training is being performed
        self.stdout.outputWritten.connect(self.append_text_evaluate)
        self.stderr.outputWritten.connect(self.append_text_evaluate)

        # Evaluate model by invoking the corresponding TM's function
        self.result_evaluation_pu_model = self.tm.evaluate_PUmodel()

        return "Done."

    def do_after_evaluate_pu_model(self):
        """
        Method to be executed after the evaluation of the PU model has been
        completed.
        """

        # Hide progress bar
        self.progress_bar_evaluate.setVisible(False)

        # Disconnect from stdout and stderr
        self.stdout.outputWritten.disconnect()
        self.stderr.outputWritten.disconnect()

        # Show results in the "table_pu_classification_results" table
        results = []
        if self.result_evaluation_pu_model is not None:
            for r, v in self.result_evaluation_pu_model.items():
                results.append(v)
            for i in np.arange(self.table_pu_classification_results.rowCount()):
                self.table_pu_classification_results.setItem(
                    i - 1, 3, QtWidgets.QTableWidgetItem(str(results[i])))
        print(results)

        self.table_pu_classification_results.resizeColumnsToContents()
        self.table_pu_classification_results.resizeRowsToContents()
        self.table_pu_classification_results.verticalHeader().setVisible(True)

        # Show informative message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The evaluation of the PU model has been completed.")

        # Show documents to annotate in 'Feedback' tab
        self.init_feedback_elements()

        return

    def clicked_evaluate_PU_model(self):
        """
        Method that controls the actions that are carried out when the button "eval_pu_classifier_push_button" is
        clicked by the user.
        """

        # Check if a corpus and a set of labels have been selected and a model trained.
        # Otherwise, the evaluation cannot be carried out
        if self.corpus_selected_name is not None and self.tm.df_labels is not None:

            # Execute the PU model evaluation in the secondary thread
            execute_in_thread(
                self, self.execute_evaluate_pu_model, self.do_after_evaluate_pu_model,
                self.progress_bar_evaluate)
        else:

            # Show warning message if the conditions for evaluating the model are not met
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.WARNING_EVALUATION)

        return

    # #########################################################################
    # GIVE FEEDBACK FUNCTIONS
    # #########################################################################
    def init_feedback_elements(self):
        """
        Method for showing the documents to be annotated in the FEEDBACK TAB.
        The number of documents that is shown depends on the value assigned to
        self.n_docs_al; the remaining widgets that exist for showing documents
        until Constants.MAX_N_DOCS are hided while they are not used.
        The widgets that represent the document are displayed as empty spaces
        until the conditions for the annotation of the documents are met, i.e.
        a corpus and a set of labels have been selected
        """

        # Hide those widgets for the annotations of the documents that are not
        # going to be used based on the value of ndocs
        for i in range(Constants.MAX_N_DOCS - self.n_docs_al):
            show_doc_name = "show_doc_" + str(self.n_docs_al + i + 1)
            show_doc_widget = self.findChild(QTextEdit, show_doc_name)
            show_doc_widget.setVisible(False)
            doc_checkbox_name = "prediction_doc_" + str(self.n_docs_al + i + 1)
            doc_checkbox_widget = self.findChild(QCheckBox, doc_checkbox_name)
            doc_checkbox_widget.setVisible(False)

        # Show docs for labeling if the conditions for it are met (i.e. a corpus and set of labels have been selected)
        if self.tm.state['selected_corpus'] and self.tm.df_labels is not None:
            self.show_sampled_docs_for_labeling()

        return

    def init_ndocs_al(self):
        """
        Initializes the AL parameter in the text edit within the third tab of the main GUI
        window, i.e. n_docs. The default configuration of this parameter is read from the
        configuration file '/config/parameters.default.yaml'.
        """

        self.text_edit_ndocs_al.setText(str(self.n_docs_al_dft))

        return

    def clicked_update_ndocs_al(self):
        """
        Updates the AL parameter that is going to be used for the resampling of the documents to be labelled based
        on the value specified by the user and shows the id, title, abstract and predicted class (if available) of each
        of the documents in a QTextEdit widget.
        """
        # Update ndocs if the user has
        if self.text_edit_ndocs_al.text():

            # Check condition for updating ndocs: only 12 documents maximum can be sampled at once
            if int(self.text_edit_ndocs_al.text()) > Constants.MAX_N_DOCS:
                message = "The number of documents to show at each AL round must be smaller than " + str(
                    Constants.MAX_N_DOCS) + ". You can choose more documents to sample at a second round."
                QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, message)
                return
            else:

                # Update ndocs
                self.n_docs_al = int(self.text_edit_ndocs_al.text())

                # Show informative message with the changes
                QtWidgets.QMessageBox.information(
                    self, Messages.DC_MESSAGE,
                    "The number of documents to show at each AL round has been set to '" + str(self.n_docs_al) + \
                    "' for the current session.")

                # Set visible the necessary widgets (QTextEdits + QCheckBoxes)
                for i in range(self.n_docs_al):
                    show_doc_name = "show_doc_" + str(i + 1)
                    show_doc_widget = self.findChild(QTextEdit, show_doc_name)
                    show_doc_widget.setVisible(True)
                    doc_checkbox_name = "prediction_doc_" + str(i + 1)
                    doc_checkbox_widget = self.findChild(QCheckBox, doc_checkbox_name)
                    doc_checkbox_widget.setVisible(True)
                    doc_checkbox_widget.setChecked(False)

        # Visualize the documents
        self.init_feedback_elements()

        return

    def show_sampled_docs_for_labeling(self):
        """
        Visualizes the documents from which the user is going to give feedback for the updating of a model.
        """
        # Select bunch of documents at random
        n_docs = self.n_docs_al
        if self.tm.dc is None or self.tm.dc.df_dataset.prediction is None:
            return

        self.selected_docs_to_annotate = self.tm.dc.AL_sample(n_samples=n_docs)

        if self.selected_docs_to_annotate is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                "There are not available predictions associated with the model of"
                "the selected labels' category. You must evaluate the model first. ")
            return

        # Indices of the selected docs
        self.idx_docs_to_annotate = self.selected_docs_to_annotate.index

        # Fill QTextWidgets with the corresponding text
        id_widget = 0
        for i, doc in self.selected_docs_to_annotate.iterrows():
            text = ""
            if self.tm.metadata['corpus_name'] == 'EU_projects':
                # Locate document in corpus
                doc_corpus = self.tm.df_corpus[self.tm.df_corpus['id'] == doc.id]
                # Get title
                title = doc_corpus.iloc[0].title
                # Get description
                descr = doc_corpus.iloc[0].description
                text_widget_name = "show_doc_" + str(id_widget + 1)
                text_widget = self.findChild(QTextEdit, text_widget_name)
                text += \
                    "<h4 style='color: #5e9ca0;'> ID" + str(doc.id) + ": " + str(title) \
                    + "</h4>\n<p>" + str(descr) + "</p>"
            else:
                descr = doc.text
                text_widget_name = "show_doc_" + str(id_widget + 1)
                text_widget = self.findChild(QTextEdit, text_widget_name)
                text += \
                    "<h4 style='color: #5e9ca0;'> ID" + str(doc.id) + ": " \
                    + "</h4>\n<p>" + str(descr) + "</p>"
            if 'prediction' in doc:
                text += "<p> <b>PREDICTED CLASS: </b>" + str(doc.prediction) + "</p>"
                doc_checkbox_name = "prediction_doc_" + str(id_widget + 1)
                doc_checkbox_widget = self.findChild(QCheckBox, doc_checkbox_name)
                if int(doc.prediction) == 1:
                    doc_checkbox_widget.setChecked(True)
                change_background_color_text_edit(text_widget, int(doc.prediction))
                self.labels_docs_to_annotate_dict[doc_checkbox_name] = int(doc.prediction)
            text_widget.setHtml(text)
            id_widget += 1

        return

    def clicked_change_predicted_class(self, checkbox):
        """
        Method to control the checking or unchecking of the QCheckboxes that represented
        the predicted class that the user has associated to each of the documents to annotate.
        """

        # Get the document widget associated to the Checkbox whose state has been changed
        doc_checkbox_widget_name = checkbox.objectName()
        text_widget_name = "show_doc_" + doc_checkbox_widget_name.split("_")[2]
        text_widget = self.findChild(QTextEdit, text_widget_name)

        # Change the color of the corresponding document widget to associate it with
        # the color of the selected predicted label
        if checkbox.isChecked():
            change_background_color_text_edit(text_widget, 1)
            self.labels_docs_to_annotate_dict[doc_checkbox_widget_name] = 1
        else:
            change_background_color_text_edit(text_widget, 0)
            self.labels_docs_to_annotate_dict[doc_checkbox_widget_name] = 0

        return

    def execute_give_feedback(self):
        """
        Method to control the annotation of a selected subset of documents based on the labels introduced by the
        user on a secondary thread while the MainWindow execution is maintained in the main thread.
        """

        # Get labels from current checkboxes
        all_labels = list(self.labels_docs_to_annotate_dict.values())
        self.labels_docs_to_annotate = all_labels[0:self.n_docs_al]

        # Call the TM function to proceed with the annotation
        self.tm.get_feedback(self.idx_docs_to_annotate, self.labels_docs_to_annotate)

        return "Done."

    def do_after_give_feedback(self):
        """
        Method to be executed after annotating the labels given by the user in their corresponding positions
        """
        # Hide progress bar
        self.progress_bar_feedback_update.setVisible(False)

        # Show message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The labels have been annotated based on your latest feedback.")

        return

    def clicked_give_feedback(self):
        """
        Method that controls the actions that are carried out when the button "give_feedback_push_button" is
        clicked by the user.
        """

        # Carry out the feedback giving in the secondary thread
        execute_in_thread(
            self, self.execute_give_feedback, self.do_after_give_feedback, self.progress_bar_feedback_update)

        return

    # #########################################################################
    # RETRAIN MODEL FUNCTIONS
    # #########################################################################
    @pyqtSlot(str)
    def append_text_retrain_reval(self, text):
        """
        Method to redirect the stdout and stderr in the "text_edit_results_reval_retrain"
        while the retraining of a PU model is being performed.
        """

        self.text_edit_results_reval_retrain.moveCursor(QTextCursor.End)
        self.text_edit_results_reval_retrain.insertPlainText(text)

        return

    def execute_retrain_model(self):
        """
        Method to control the execution of the retraining of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """

        # Connect to stdout and stderr
        self.stdout.outputWritten.connect(self.append_text_retrain_reval)
        self.stderr.outputWritten.connect(self.append_text_retrain_reval)

        # Retrain the PU model by invoking the task manager method
        self.tm.retrain_model()

        return

    def do_after_retrain_model(self):
        """
        Method to be executed once the retraining of the model based on the feedback of the user has been completed.
        """
        # Hide progress bar
        self.progress_bar_feedback_update.setVisible(False)

        # Disconnect from stdout and stderr
        self.stdout.outputWritten.disconnect()
        self.stderr.outputWritten.disconnect()

        # Show message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The performance of the classifier model has been improved using the labels you provided.")

        return

    def clicked_retrain_model(self):
        """
        Method that controls the actions that are carried out when the button "retrain_model_push_button" is
        clicked by the user.
        """

        # Check if a corpus has been selected. Otherwise, the retraining cannot be carried out
        if self.corpus_selected_name is not None and self.tm.df_labels is not None:

            # Execute the PU model retraining in the secondary thread
            execute_in_thread(
                self, self.execute_retrain_model, self.do_after_retrain_model, self.progress_bar_feedback_update)
        else:

            # Show warning message if retraining could not be carried out as either no corpus or
            # not labels were selected at the time the request was made
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.WARNING_RETRAINING)

        return

    # #########################################################################
    # REEVALUATE MODEL FUNCTIONS
    # #########################################################################
    def execute_reevaluate_model(self):
        """
        Method to control the execution of the reevaluation of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """

        # Connect to stdout and stderr
        self.stdout.outputWritten.connect(self.append_text_retrain_reval)
        self.stderr.outputWritten.connect(self.append_text_retrain_reval)

        # Reevaluate model by invoking the correspinding TM's method
        self.result_reevaluation_pu_model = self.tm.reevaluate_model()

        return "Done."

    def do_after_reevaluate_model(self):
        """
        Method to be executed once the reevaluation of the model based on the feedback of the user has been completed.
        """

        # Hide progress bar
        self.progress_bar_feedback_update.setVisible(False)

        # Disconnect from stdout and stderr
        self.stdout.outputWritten.disconnect()
        self.stderr.outputWritten.disconnect()

        # Show results in the "table_pu_classification_results" table
        if self.result_reevaluation_pu_model is not None:
            results = []
            for r, v in self.result_reevaluation_pu_model.items():
                results.append(v)
            for i in np.arange(self.table_reclassification_results.rowCount()):
                self.table_reclassification_results.setItem(
                    i - 1, 3, QtWidgets.QTableWidgetItem(str(results[i])))

        self.table_reclassification_results.resizeColumnsToContents()
        self.table_reclassification_results.resizeRowsToContents()
        self.table_pu_classification_results.verticalHeader().setVisible(True)

        # Show message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The reevaluation of the classifier model has been completed.")

        return

    def clicked_reevaluate_model(self):
        """
        Method that controls the actions that are carried out when the button "retrain_model_push_button" is
        clicked by the user.
        """

        # Check if a corpus and a set of labels have been selected and a model trained.
        # Otherwise, the reeevaluation cannot be carried out
        if self.corpus_selected_name is not None and self.tm.df_labels is not None:

            # Execute the PU model reevaluation in the secondary thread
            execute_in_thread(
                self, self.execute_reevaluate_model, self.do_after_reevaluate_model, self.progress_bar_feedback_update)
        else:

            # Show warning message if the conditions for evaluating the model are not met
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.WARNING_REEVALUATION)
        return
