# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                         CLASS MAIN WINDOW                              ***
******************************************************************************
Class representing the main window of the application.

"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
# General imports
import numpy as np
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup, QDesktopWidget, QTextEdit
from PyQt5.QtCore import QThreadPool, QSize
from PyQt5.QtGui import QPixmap

# Local imports
from src.graphical_user_interface.analyze_keywords_window import (
    AnalyzeKeywordsWindow)
from src.graphical_user_interface.get_keywords_window import GetKeywordsWindow
from src.graphical_user_interface.get_topics_list_window import (
    GetTopicsListWindow)
from src.graphical_user_interface.messages import Messages
from src.graphical_user_interface.util import toggle_menu, execute_in_thread
from src.graphical_user_interface.constants import Constants


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, project_folder, source_folder, tm, widget):
        super(MainWindow, self).__init__()

        # Load UI and configure default geometry of the window
        #######################################################################
        uic.loadUi("UIS/DomainClassifier.ui", self)
        self.init_ui()
        self.animation = QtCore.QPropertyAnimation(self.frame_left_menu,
                                                   b"minimumWidth")

        # ATTRIBUTES
        #######################################################################
        self.source_folder = source_folder
        self.project_folder = project_folder
        self.tm = tm
        self.widget = widget
        self.corpus_selected_name = ""
        self.labels_loaded = None
        self.get_label_option = 0
        self.message_out = None
        self.get_keywords_window = GetKeywordsWindow(tm)
        self.analyze_keywords_window = AnalyzeKeywordsWindow(tm)
        self.get_topics_list_window = GetTopicsListWindow(tm)
        self.class_max_imbalance_dft = self.tm.global_parameters['classifier']['max_imbalance']
        self.class_nmax_dft = self.tm.global_parameters['classifier']['nmax']
        self.class_max_imbalance = self.class_max_imbalance_dft
        self.class_nmax = self.class_nmax_dft
        self.result_evaluation_pu_model = None
        self.text_to_print_train = ""
        self.text_to_print_eval = ""
        self.n_docs_al_dft = self.tm.global_parameters['active_learning']['n_docs']
        self.n_docs_al = self.n_docs_al_dft
        self.selected_docs_to_annotate = None
        self.idx_docs_to_annotate = None
        self.labels_docs_to_annotate = []

        # INFORMATION BUTTONS: Set image, size and tooltip
        # #####################################################################
        # TAB LOADING
        self.info_button_select_corpus.setIcon(QIcon('Images/help2.png'))
        self.info_button_select_corpus.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_select_corpus.width(),
                                            self.info_button_select_corpus.height()))
        self.info_button_select_corpus.setToolTip(Messages.INFO_SELECT_CORPUS)
        self.info_button_get_labels.setIcon(QIcon('Images/help2.png'))
        self.info_button_get_labels.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_get_labels.width(),
                                            self.info_button_get_labels.height()))
        self.info_button_get_labels.setToolTip(Messages.INFO_GET_LABELS)
        self.info_button_load_reset_labels.setIcon(QIcon('Images/help2.png'))
        self.info_button_load_reset_labels.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_load_reset_labels.width(),
                                            self.info_button_load_reset_labels.height()))
        self.info_button_load_reset_labels.setToolTip(
            Messages.INFO_LOAD_RESET_LABELS)
        # TAB TRAINING-EVALUATION
        self.info_button_train_model.setIcon(QIcon('Images/help2.png'))
        self.info_button_train_model.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_load_reset_labels.width(),
                                            self.info_button_load_reset_labels.height()))
        self.info_button_train_model.setToolTip(Messages.INFO_TRAIN_PU_MODEL)
        self.info_button_eval_model.setIcon(QIcon('Images/help2.png'))
        self.info_button_eval_model.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_load_reset_labels.width(),
                                            self.info_button_load_reset_labels.height()))
        self.info_button_eval_model.setToolTip(Messages.INFO_EVALUATE_PU_MODEL)
        # TAB FEEDBACK
        self.info_button_give_feedback.setIcon(QIcon('Images/help2.png'))
        self.info_button_give_feedback.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_load_reset_labels.width(),
                                            self.info_button_load_reset_labels.height()))
        self.info_button_give_feedback.setToolTip(Messages.INFO_FEEDBACK)
        self.info_button_ndocs_al.setIcon(QIcon('Images/help2.png'))
        self.info_button_ndocs_al.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_load_reset_labels.width(),
                                            self.info_button_load_reset_labels.height()))
        self.info_button_ndocs_al.setToolTip(Messages.INFO_N_DOCS_AL)

        # #####################################################################
        # CONFIGURE ELEMENTS IN THE "LOAD CORPUS VIEW"
        # #####################################################################
        self.progress_bar_first_tab.setVisible(False)
        self.progress_bar_first_tab.setValue(0)

        # LOAD CORPUS WIDGETS
        # #####################################################################
        self.load_corpus_push_button.clicked.connect(self.clicked_load_corpus)
        self.show_corpora()

        # GET LABELS WIDGETS
        # #####################################################################
        self.get_labels_radio_buttons = QButtonGroup(self)
        self.get_labels_radio_buttons.addButton(self.get_labels_option1, 1)
        self.get_labels_radio_buttons.addButton(self.get_labels_option2, 2)
        self.get_labels_radio_buttons.addButton(self.get_labels_option3, 3)
        self.get_labels_radio_buttons.addButton(self.get_labels_option4, 4)
        self.get_labels_radio_buttons.addButton(self.get_labels_option5, 5)
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
        self.train_model_push_button.clicked.connect(
            self.clicked_train_PU_model)
        self.progress_bar_train_evaluate_pu.setVisible(False)
        self.progress_bar_train_evaluate_pu.setValue(0)
        self.eval_model_push_button.clicked.connect(
            self.clicked_evaluate_PU_model)
        self.update_params_model_push_button.clicked.connect(
            self.update_params_train_pu_model)
        self.table_params_train_model_pu.setToolTip(Messages.INFO_TABLE_TRAIN_PU_MODEL)

        self.progress_bar_train_evaluate_pu.setVisible(False)
        self.progress_bar_train_evaluate_pu.setValue(0)

        self.init_params_train_pu_model()

        # GET FEEDBACK WIDGETS
        # #####################################################################
        self.update_model_push_button.clicked.connect(
            self.clicked_update_model)
        self.table_labels_feedback.cellChanged.connect(
            self.clicked_labels_from_docs_updated)
        self.update_ndocs_al_push_button.clicked.connect(self.clicked_update_ndocs_al)
        self.clear_feedback_push_button.clicked.connect(self.clicked_clear_feedback)
        self.get_docs_to_annotate_push_button.clicked.connect(
            self.show_sampled_docs_for_labeling)

        self.progress_bar_feedback.setVisible(False)
        self.progress_bar_feedback.setValue(0)

        self.init_ndocs_al()
        self.init_feedback_table(False)

        # THREADS FOR EXECUTING IN PARALLEL
        # #####################################################################
        self.thread_pool = QThreadPool()
        print("Multithreading with maximum"
              " %d threads" % self.thread_pool.maxThreadCount())
        self.worker = None
        self.loading_window = None

        # TOGGLE MENU
        # #####################################################################
        self.toggleButton.clicked.connect(lambda: toggle_menu(self, 250))
        self.toggleButton.setIcon(QIcon('Images/menu.png'))
        self.toggleButton.setIconSize(Constants.BUTTONS_SCALE * QSize(self.toggleButton.width(),
                                                                      self.toggleButton.height()))

        # PAGES
        # #####################################################################
        # PAGE 1: Load corpus/ labels
        self.pushButtonLoad.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_load))
        self.pushButtonLoad.setIcon(QIcon('Images/settings.png'))
        self.pushButtonLoad.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.pushButtonLoad.width(),
                                            self.pushButtonLoad.height()))
        self.pushButtonLoad.setToolTip(Messages.INFO_LOAD_CORPUS_LABELS)
        # PAGE 2: Train classifier
        self.pushButtonTrain.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_train))
        self.pushButtonTrain.setIcon(QIcon('Images/training.png'))
        self.pushButtonTrain.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.pushButtonTrain.width(),
                                            self.pushButtonTrain.height()))
        self.pushButtonTrain.setToolTip(Messages.INFO_TRAIN_EVALUATE_PU)
        # PAGE 3: Get relevance feedback
        self.pushButtonGetFeedback.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_feedback))
        self.pushButtonGetFeedback.setIcon(QIcon('Images/feedback.png'))
        self.pushButtonGetFeedback.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.pushButtonTrain.width(),
                                            self.pushButtonTrain.height()))
        self.pushButtonGetFeedback.setToolTip(Messages.INFO_GET_FEEDBACK)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        pixmap = QPixmap('Images/dc_logo2.png')
        self.label_logo.setPixmap(pixmap)
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.resize(self.minimumSizeHint())
        self.center()

    def center(self):
        """Centers the window at the middle of the screen at which the application is being executed.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # #################################################################################################################
    # LOAD CORPUS FUNCTIONS
    # #################################################################################################################
    def show_corpora(self):
        """ List all corpora contained in the source folder selected by the user.
        """
        corpus_list = self.tm.DM.get_corpus_list()
        for corpus_nr in np.arange(0, len(corpus_list), 1):
            self.tree_view_select_corpus.insertItem(
                corpus_nr, corpus_list[corpus_nr])

        return

    def execute_load_corpus(self):
        """ Method to control the execution of the loading of a corpus on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """
        self.tm.load_corpus(self.corpus_selected_name)
        return "Done."

    def do_after_load_corpus(self):
        """ Method to be executed after the loading of the corpus has been
        completed.
        """
        # Hide progress bar
        self.progress_bar_first_tab.setVisible(False)

        # Showing messages in the status bar, pop up window, and corpus label
        self.statusBar().showMessage(
            "'" + self.corpus_selected_name + "' was selected as corpus.",
            Constants.LONG_TIME_SHOW_SB)
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The corpus '" + self.corpus_selected_name + "' has been "
                                                         "loaded in the current session.")

        self.label_corpus_selected_is.setText(str(self.corpus_selected_name))
        self.show_labels()

    def clicked_load_corpus(self):
        """ Method to control the selection of a new corpus by double-clicking
        into one of the items of the corpus list
        within the selected source folder, as well as its loading as dataframe
        into the TaskManager object.
        Important is that the corpus cannot be changed inside the same project,
        so if a corpus was used before me must keep the same one.
        """
        item = self.tree_view_select_corpus.currentItem()
        corpus_name = str(item.text())
        current_corpus = self.tm.metadata['corpus_name']

        # Go back to the main window of the application so the user can select a different project folder in case he
        # chooses a different corpus for an already existing project
        if self.tm.state['selected_corpus'] and corpus_name != current_corpus:
            warning = "The corpus of this project is " + current_corpus + \
                      ". Run another project to use " + corpus_name + "."
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, warning)
            self.widget.removeWidget(self.widget.currentWidget())
            return

        self.corpus_selected_name = corpus_name

        # Load corpus into the TaskManager object as dataframe
        execute_in_thread(
            self, self.execute_load_corpus, self.do_after_load_corpus, self.progress_bar_first_tab)

    # #########################################################################
    # GET LABELS FUNCTIONS
    # #########################################################################
    def execute_import_labels(self):
        """Imports the labels by invoking the corresponding method in the Task Manager object associated with the GUI.
        """
        # Get labels
        if self.get_label_option == 1:
            message_out = self.tm.import_labels()
        elif self.get_label_option == 2:
            message_out = self.tm.get_labels_by_keywords(
                self.get_keywords_window.selectedKeywords,
                self.get_keywords_window.selectedTag)
        elif self.get_label_option == 3:
            message_out = self.tm.get_labels_by_keywords(
                self.get_keywords_window.selectedKeywords,
                self.get_keywords_window.selectedTag)
        elif self.get_label_option == 4:
            message_out = self.tm.get_labels_by_topics(
                topic_weights=self.get_topics_list_window.tw,
                T=self.get_topics_list_window.T,
                df_metadata=self.get_topics_list_window.df_metadata,
                n_max=self.get_topics_list_window.n_max,
                s_min=self.get_topics_list_window.s_min,
                tag=self.get_topics_list_window.selectedTag)
        elif self.get_label_option == 5:
            df_labels, message_out = self.tm.get_labels_by_definitions()

        self.message_out = message_out[3:]
        return "Done"

    def do_after_import_labels(self):
        """Function to be executed after the labels' importing has been completed.
        """
        # Hide progress bar
        self.progress_bar_first_tab.setVisible(False)
        # Show informative message
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE, self.message_out)

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
                # @ TODO: To be implemented
        return

    def clicked_get_labels(self):
        """Method for performing the getting of the labels according to the
        method selected for it.
        """
        if self.get_label_option == 0:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_LABEL_OPTION_SELECTED)
        else:
            if self.get_label_option == 1:
                execute_in_thread(self, self.execute_import_labels,
                                  self.do_after_import_labels, self.progress_bar_first_tab)
            elif self.get_label_option == 2:
                execute_in_thread(self, self.execute_import_labels,
                                  self.do_after_import_labels, self.progress_bar_first_tab)
            elif self.get_label_option == 3:
                execute_in_thread(self, self.execute_import_labels,
                                  self.do_after_import_labels, self.progress_bar_first_tab)
                # Show the window for the analysis of the keywords
                self.analyze_keywords_window.do_analysis()
                self.analyze_keywords_window.exec()
            elif self.get_label_option == 4:
                execute_in_thread(self, self.execute_import_labels,
                                  self.do_after_import_labels, self.progress_bar_first_tab)
            elif self.get_label_option == 5:
                execute_in_thread(self, self.execute_import_labels,
                                  self.do_after_import_labels, self.progress_bar_first_tab)

        # Load just gotten labels
        self.show_labels()

        # Reset after loading labels
        self.get_labels_radio_buttons.setExclusive(False)
        self.get_labels_option1.setChecked(False)
        self.get_labels_option2.setChecked(False)
        self.get_labels_option3.setChecked(False)
        self.get_labels_option4.setChecked(False)
        self.get_labels_option5.setChecked(False)
        self.get_labels_radio_buttons.setExclusive(True)
        self.get_label_option == 0
        return

    # #########################################################################
    # LOAD LABELS FUNCTIONS
    # #########################################################################
    def show_labels(self):
        """ Method for showing the labels associated with the selected corpus.
        """
        self.tree_view_load_labels.clear()
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            labelset_list = self.tm.DM.get_labelset_list(
                self.corpus_selected_name)
            for labelset_nr in np.arange(0, len(labelset_list), 1):
                self.tree_view_load_labels.insertItem(
                    labelset_nr, labelset_list[labelset_nr])

    def clicked_load_labels(self):
        """Method for controlling the loading of the labels into the session.
        It is equivalent to the "_get_labelset_list" method from the
        TaskManager class
        """
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            item = self.tree_view_load_labels.currentItem()
            self.labels_loaded = str(item.text())
            self.tm.load_labels(self.labels_loaded)

            # Showing messages in the status bar, pop up window, and corpus
            # label
            self.statusBar().showMessage(
                "'" + self.labels_loaded + "' were loaded.", Constants.LONG_TIME_SHOW_SB)
            QtWidgets.QMessageBox.information(
                self, Messages.DC_MESSAGE,
                "The labels '" + self.labels_loaded + "' have been loaded"
                                                      " in the current session.")

            self.label_labels_loaded_are.setText(str(self.labels_loaded))

        return

    def clicked_reset_labels(self):
        """
        Method for controlling the resetting of the current session's labels.
        """
        item = self.tree_view_load_labels.currentItem()
        self.labels_loaded = str(item.text())
        if self.tm.CorpusProc is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            if self.labels_loaded is not None:
                self.tm.reset_labels(self.labels_loaded)
            aux_labels = self.labels_loaded
            self.labels_loaded = None
            # Showing messages in the status bar, pop up window, and corpus
            # label
            self.statusBar().showMessage(
                "'" + aux_labels + "' were removed.", Constants.LONG_TIME_SHOW_SB)
            QtWidgets.QMessageBox.information(
                self, Messages.DC_MESSAGE,
                "The labels '" + aux_labels + "' have been removed from the "
                                              "current session.")

            self.label_labels_loaded_are.setText(str(" "))
            self.show_labels()
        return

    # #########################################################################
    # TRAIN PU MODEL FUNCTIONS
    # #########################################################################
    def init_params_train_pu_model(self):
        """Initializes the classifier parameters in the parameters' table within the second tab of the main GUI
        window, i.e. max_imbalance and nmax. The default configuration of these parameters is read from the
        configuration file '/config/parameters.default.yaml'.
        """
        self.table_params_train_model_pu.clearContents()
        self.table_params_train_model_pu.setRowCount(1)
        self.table_params_train_model_pu.setColumnCount(2)

        self.table_params_train_model_pu.setItem(
            0, 0, QtWidgets.QTableWidgetItem(str(self.class_max_imbalance)))
        self.table_params_train_model_pu.setItem(
            0, 1, QtWidgets.QTableWidgetItem(str(self.class_nmax)))

    def update_params_train_pu_model(self):
        """Updates the classifier parameters that are going to be used for the training of the PU model based on the
        values read from the table within the second tab of the main GUI
        window that have been specified by the user.
        """
        if self.table_params_train_model_pu.item(0, 0) is not None:
            self.class_max_imbalance = int(self.table_params_train_model_pu.item(0, 0).text())
        else:
            self.class_max_imbalance = self.class_max_imbalance_dft

        if self.table_params_train_model_pu.item(0, 1) is not None:
            self.class_nmax = int(self.table_params_train_model_pu.item(0, 1).text())
        else:
            self.class_nmax = self.class_nmax_dft

        self.init_params_train_pu_model()

        # Show informative message with the changes
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The parameters of classifier has been set to '" + str(self.class_max_imbalance) + "' and '" + \
            str(self.class_nmax) + "' for the current session.")

    def execute_train_classifier(self):
        """Method to control the execution of the training of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """
        # Train the PU model by invoking the task manager method
        self.tm.train_PUmodel()

        # Get results of the training by reading the log file
        p = self.tm.global_parameters['logformat']
        file_to_follow = self.tm.path2project / p['filename']
        self.text_to_print_train = "<h1 style='color: #5e9ca0;'> PU MODEL TRAINING RESULTS: </h1>\n<li/>"
        with open(file_to_follow) as file:
            for line in (file.readlines()[-Constants.READ_LAST_LOGS:]):
                self.text_to_print_train += line[:-1] + "</li><li/>"
        self.text_to_print_train = self.text_to_print_train[:-5]
        return "Done."

    def do_after_train_classifier(self):
        """ Method to be executed after the training of the classifier has been
        completed.
        """
        # Hide progress bar
        self.progress_bar_train_evaluate_pu.setVisible(False)

        # Show logs in the QTextEdit
        self.text_logs_training.setHtml(self.text_to_print_train)

        # Showing message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The PU model has been trained.")

    def clicked_train_PU_model(self):
        """ Method that control the actions that are carried out when the button "train_model_push_button" is
        clicked by the user.
        """
        # Check if a corpus has been selected. Otherwise, the training cannot be carried out
        if self.corpus_selected_name is not None:
            # Execute the PU model training in the secondary thread
            execute_in_thread(
                self, self.execute_train_classifier, self.do_after_train_classifier, self.progress_bar_train_evaluate_pu)
        else:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.WARNING_TRAINING)
        return

    # #########################################################################
    # EVALUATE PU MODEL FUNCTIONS
    # #########################################################################
    def execute_evaluate_pu_model(self):
        """Method to control the execution of the training of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """
        self.result_evaluation_pu_model = self.tm.evaluate_PUmodel()
        return "Done."

    def do_after_evaluate_pu_model(self):
        """
        Method to be executed after the evaluation of the PU model has been
        completed.
        """
        # Hide progress bar
        self.progress_bar_train_evaluate_pu.setVisible(False)

        # Show results in the QTextEdit
        if self.result_evaluation_pu_model is not None:
            self.text_to_print_eval = "<h1 style='color:#5e9ca0;'> CLASSIFICATION RESULTS: </h1><ul>"
            for r, v in self.result_evaluation_pu_model.items():
                self.text_to_print_eval += "<li><b>" + str(r) + ":</b> " + str(v) + "</li>"
            self.text_to_print_eval += "</ul>"
            self.text_edit_results_eval_model.setText(self.text_to_print_eval)

        # Show informative message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The evaluation of the PU model has been completed.")

    def clicked_evaluate_PU_model(self):
        """ Method that control the actions that are carried out when the button "eval_model_push_button" is
        clicked by the user.
        """
        # Check if a corpus has been selected. Otherwise, the evaluation cannot be carried out
        if self.corpus_selected_name is not None:
            # Execute the PU model evaluation in the secondary thread
            execute_in_thread(
                self, self.execute_evaluate_pu_model, self.do_after_evaluate_pu_model, self.progress_bar_train_evaluate_pu)
        else:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.WARNING_EVALUATION)
        return

    # #########################################################################
    # GET FEEDBACK FUNCTIONS
    # #########################################################################
    def init_ndocs_al(self):
        """Initializes the AL parameter in the text edit within the third tab of the main GUI
        window, i.e. n_docs. The default configuration of this parameter is read from the
        configuration file '/config/parameters.default.yaml'.
        """
        self.text_edit_ndocs_al.setText(str(self.n_docs_al_dft))

    def clicked_update_ndocs_al(self):
        """Updates the AL parameter that is going to be used based on the value specified by the user.
        """
        # Update ndocs if possible
        if self.text_edit_ndocs_al.text():
            selected_docs = self.tm.dc.df_dataset.loc[
                self.df_dataset.prediction != Constants.UNUSED]
            # Check condition for updating ndocs
            if len(selected_docs) < int(self.text_edit_ndocs_al.text()):
                message = "The number of documents to show at each AL round must be smaller than " + str(len(selected_docs))
                QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, message)
                return
            else:
                self.n_docs_al = int(self.text_edit_ndocs_al.text())
                # Show informative message with the changes
                QtWidgets.QMessageBox.information(
                    self, Messages.DC_MESSAGE,
                    "The number of documents to show at each AL round has been set to '" + str(self.n_docs_al) + \
                    "' for the current session.")

    def init_feedback_table(self, visibility):
        """Sets the initial settings of table "table_labels_feedback" in which the user is going to specify the
        labels for the showed documents in the feedback tab.
        """
        self.table_labels_feedback.clearContents()
        self.table_labels_feedback.setColumnCount(2)
        self.table_labels_feedback.setVisible(visibility)

    def show_sampled_docs_for_labeling(self):
        """Visualizes the documents from which the user is going to give feedback for the updating of a model.
        """
        # Select bunch of documents at random
        n_docs = self.n_docs_al
        self.selected_docs_to_annotate = self.tm.dc.AL_sample(n_samples=n_docs)
        # Indices of the selected docs
        self.idx_docs_to_annotate = self.selected_docs_to_annotate.index

        # Add necessary widgets to represent each of the sampled docs in case more than N_DOCS needs to be displayed
        if len(self.selected_docs_to_annotate) > int(Constants.N_DOCS):
            widgets_to_add = Constants.N_DOCS - len(self.selected_docs_to_annotate)
            label = int(Constants.N_DOCS)
            for i in np.arange(widgets_to_add):
                new_show_doc = QTextEdit()
                new_show_doc_name = "show_doc_" + str(label + i)
                new_show_doc.setObjectName(new_show_doc_name)
                self.grid_layout_docs_feedback.addWidget(new_show_doc)

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
                    "<h3 style='color: #5e9ca0;'> ID" + str(doc.id) + ": " + str(title) \
                    + "</h3>\n<p>" + str(descr) + "</p>"
            else:
                descr = doc.text
                text_widget_name = "show_doc_" + str(id_widget + 1)
                text_widget = self.findChild(QTextEdit, text_widget_name)
                text += \
                    "<h3 style='color: #5e9ca0;'> ID" + str(doc.id) + ": " \
                    + "</h3>\n<p>" + str(descr) + "</p>"
            if 'prediction' in doc:
                text += "<p> <b>PREDICTED CLASS: </b>" + str(doc.prediction) + "</p>"
            text_widget.setHtml(text)
            self.table_labels_feedback.setItem(
                id_widget, 0, QtWidgets.QTableWidgetItem(str(doc.id)))
            id_widget += 1

        # Configure "table_labels_feedback" with one row per "selected_doc" and make it visible
        self.clicked_clear_feedback()

    def clicked_labels_from_docs_updated(self):
        """Gets labels that are going to be used for the updating of the model.
        """
        labels = []
        for i in np.arange(self.table_labels_feedback.rowCount()):
            if self.table_labels_feedback.item(i, 1) is not None:
                labels_doc = str(
                    self.table_labels_feedback.item(i, 0).text())
                labels.append(labels_doc)
        print(labels)
        self.labels_docs_to_annotate = labels

    def execute_give_feedback(self):
        """Method to control the annotation of a selected subset of documents based on the labels introduced by the
        user on a secondary thread while the MainWindow execution is maintained in the main thread.
        """
        self.tm.get_feedback(self.idx_docs_to_annotate, self.labels_docs_to_annotate)
        return "Done."

    def do_after_give_feedback(self):
        """ Method to be executed after the updating of the model based on the feedback of the user has been completed.
        """
        # Hide progress bar
        self.progress_bar_feedback.setVisible(False)

        # Showing message in pop up window
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The classifier model has been updated with the latest relevance feedback.")

    def clicked_update_model(self):
        """Method that control the actions that are carried out when the button "update_model_push_button" is
        clicked by the user.
        """
        execute_in_thread(
            self, self.execute_give_feedback, self.do_after_give_feedback, self.progress_bar_feedback)
        return

    def clicked_clear_feedback(self):
        """Method that control the actions that are carried out when the button "clear_feedback_push_button" is
        clicked by the user.
        """
        self.init_feedback_table(True)
        self.table_labels_feedback.setRowCount(len(self.selected_docs_to_annotate))
        id_row = 0
        for i, doc in self.selected_docs_to_annotate.iterrows():
            self.table_labels_feedback.setItem(
                id_row, 0, QtWidgets.QTableWidgetItem(str(doc.id)))
            id_row += 1
