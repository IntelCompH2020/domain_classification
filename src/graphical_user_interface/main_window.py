# -*- coding: utf-8 -*-
"""
Class representing the main window of the application.

@author: lcalv
"""

import numpy as np

from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup, QDesktopWidget
from PyQt5.QtCore import QThreadPool, QSize
from PyQt5.QtGui import QPixmap

from src.graphical_user_interface.analyze_keywords_window import (
    AnalyzeKeywordsWindow)
from src.graphical_user_interface.get_keywords_window import GetKeywordsWindow
from src.graphical_user_interface.get_topics_list_window import (
    GetTopicsListWindow)
from src.graphical_user_interface.messages import Messages
from src.graphical_user_interface.util import toggle_menu, execute_in_thread, follow

# CONSTANTS
BUTTONS_SCALE = 0.75


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

        # INFORMATION BUTTONS
        # #####################################################################
        self.info_button_select_corpus.setIcon(QIcon('Images/help2.png'))
        self.info_button_select_corpus.setIconSize(
            BUTTONS_SCALE * QSize(self.info_button_select_corpus.width(),
                                  self.info_button_select_corpus.height()))
        self.info_button_select_corpus.setToolTip(Messages.INFO_SELECT_CORPUS)
        self.info_button_get_labels.setIcon(QIcon('Images/help2.png'))
        self.info_button_get_labels.setIconSize(
            BUTTONS_SCALE * QSize(self.info_button_get_labels.width(),
                                  self.info_button_get_labels.height()))
        self.info_button_get_labels.setToolTip(Messages.INFO_GET_LABELS)
        self.info_button_load_reset_labels.setIcon(QIcon('Images/help2.png'))
        self.info_button_load_reset_labels.setIconSize(
            BUTTONS_SCALE * QSize(self.info_button_load_reset_labels.width(),
                                  self.info_button_load_reset_labels.height()))
        self.info_button_load_reset_labels.setToolTip(
            Messages.INFO_LOAD_RESET_LABELS)

        # #####################################################################
        # CONFIGURE ELEMENTS IN THE "LOAD CORPUS VIEW"
        # #####################################################################

        self.progress_bar_first_window.setVisible(False)
        self.progress_bar_first_window.setValue(0)

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
        self.train_pu_model_push_button.clicked.connect(
            self.clicked_train_PU_model)
        self.progress_bar_train.setVisible(False)
        self.progress_bar_train.setValue(0)

        self.evaluate_pu_model_push_button.clicked.connect(
            self.clicked_evaluate_PU_model)

        # GET FEEDBACK WIDGETS
        # #####################################################################
        # @ TODO: Change
        self.train_pu_model_push_button.clicked.connect(
            self.clicked_train_PU_model)

        # GET FEEDBACK WIDGETS
        # #####################################################################
        self.update_model_push_button.clicked.connect(
            self.clicked_update_model)

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
        self.toggleButton.setIconSize(BUTTONS_SCALE * QSize(self.toggleButton.width(),
                                                            self.toggleButton.height()))

        # PAGES
        # #####################################################################
        # PAGE 1: Load corpus/ labels
        self.pushButtonLoad.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_load))
        self.pushButtonLoad.setIcon(QIcon('Images/settings.png'))
        self.pushButtonLoad.setIconSize(
            BUTTONS_SCALE * QSize(self.pushButtonLoad.width(),
                                  self.pushButtonLoad.height()))
        self.pushButtonLoad.setToolTip(Messages.INFO_LOAD_CORPUS_LABELS)
        # PAGE 2: Train classifier
        self.pushButtonTrain.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_train))
        self.pushButtonTrain.setIcon(QIcon('Images/training.png'))
        self.pushButtonTrain.setIconSize(
            BUTTONS_SCALE * QSize(self.pushButtonTrain.width(),
                                  self.pushButtonTrain.height()))
        self.pushButtonTrain.setToolTip(Messages.INFO_TRAIN_CLASSIFIER)
        # PAGE 3: Get relevance feedback
        self.pushButtonGetFeedback.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.page_feedback))
        self.pushButtonGetFeedback.setIcon(QIcon('Images/feedback.png'))
        self.pushButtonGetFeedback.setIconSize(
            BUTTONS_SCALE * QSize(self.pushButtonTrain.width(),
                                  self.pushButtonTrain.height()))
        self.pushButtonGetFeedback.setToolTip(Messages.INFO_GET_FEEDBACK)

    def init_ui(self):
        # Update image
        pixmap = QPixmap('Images/dc_logo2.png')
        self.label_logo.setPixmap(pixmap)
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.resize(self.minimumSizeHint())
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # #################################################################################################################
    # LOAD CORPUS FUNCTIONS
    # #################################################################################################################
    def show_corpora(self):
        """
        List all corpora contained in the source folder selected by the user
        """
        corpus_list = self.tm.DM.get_corpus_list()
        for corpus_nr in np.arange(0, len(corpus_list), 1):
            self.tree_view_select_corpus.insertItem(
                corpus_nr, corpus_list[corpus_nr])

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
        self.progress_bar_first_window.setVisible(False)

        # Showing messages in the status bar, pop up window, and corpus label
        self.statusBar().showMessage(
            "'" + self.corpus_selected_name + "' was selected as corpus.",
            10000)
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The corpus '" + self.corpus_selected_name + "' has been "
                                                         "loaded in the current session.")

        self.label_corpus_selected_is.setText(str(self.corpus_selected_name))
        self.show_labels()

    def clicked_load_corpus(self):
        """
        Method to control the selection of a new corpus by double-clicking
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
        self.statusBar().showMessage(
            "'The corpus " + self.corpus_selected_name + "' is being loaded.",
            3000)
        execute_in_thread(
            self, self.execute_load_corpus, self.do_after_load_corpus, self.progress_bar_first_window)

    # #########################################################################
    # GET LABELS FUNCTIONS
    # #########################################################################
    def execute_import_labels(self):
        message_out = self.tm.import_labels()
        self.message_out = message_out
        return "Done"

    def do_after_import_labels(self):
        # Hide progress bar
        self.progress_bar_first_window.setVisible(False)

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
        else:
            if self.get_label_option == 1:
                self.statusBar().showMessage(
                    "Labels are being loaded from source file.", 5000)
                execute_in_thread(self, self.execute_import_labels,
                                  self.do_after_import_labels, self.progress_bar_first_window)
            elif self.get_label_option == 2:
                message_out = self.tm.get_labels_by_keywords(
                    self.get_keywords_window.selectedKeywords,
                    self.get_keywords_window.selectedTag)
            elif self.get_label_option == 3:
                message_out = self.tm.get_labels_by_keywords(
                    self.get_keywords_window.selectedKeywords,
                    self.get_keywords_window.selectedTag)
                # Show the window for the analysis of the keywords
                self.analyze_keywords_window.do_analysis()
                self.analyze_keywords_window.exec()
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

            if self.get_label_option != 1 and self.get_label_option != -1:
                self.message_out = message_out
                self.do_after_import_labels()

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
        """
        Method for showing the labels associated with the selected corpus.
        """
        self.tree_view_load_labels.clear()
        if self.corpus_selected_name is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_NO_CORPUS_SELECTED)
        else:
            labelset_list = self.tm.DM.get_labelset_list(self.corpus_selected_name)
            for labelset_nr in np.arange(0, len(labelset_list), 1):
                self.tree_view_load_labels.insertItem(
                    labelset_nr, labelset_list[labelset_nr])

    def clicked_load_labels(self):
        """
        Method for controlling the loading of the labels into the session.
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
                "'" + self.labels_loaded + "' were loaded.", 10000)
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
                "'" + aux_labels + "' were removed.", 10000)
            QtWidgets.QMessageBox.information(
                self, Messages.DC_MESSAGE,
                "The labels '" + aux_labels + "' have been removed from the "
                                              "current session.")

            self.label_labels_loaded_are.setText(str(" "))
            self.show_labels()
        return

    # #########################################################################
    # TRAIN CLASSIFIER FUNCTIONS
    # #########################################################################
    def execute_train_classifier(self):
        """
        Method to control the execution of the training of a classifier on a
        secondary thread while the MainWindow execution is maintained in the
        main thread.
        """
        self.tm.train_PUmodel()
        return "Done."

    def do_after_train_classifier(self):
        """
        Method to be executed after the training of the classifier has been
        completed.
        """
        # Hide progress bar
        self.progress_bar_train.setVisible(False)

        # Show logs in the QTextEdit
        logs_training = follow(self)
        for log in logs_training:
            while "-- Loading PU dataset" not in log:
                continue
            self.text_edit_logs_training.setText(log)

        # Showing messages in the status bar, pop up window, and corpus label
        QtWidgets.QMessageBox.information(
            self, Messages.DC_MESSAGE,
            "The classifier was trained.")

        self.label_corpus_selected_is.setText(str(self.corpus_selected_name))
        self.show_labels()

    def clicked_train_PU_model(self):
        execute_in_thread(
            self, self.execute_train_classifier, self.do_after_train_classifier, self.progress_bar_train)

        return

    def clicked_evaluate_PU_model(self):
        print("TODO")

    # #########################################################################
    # GET FEEDBACK FUNCTIONS
    # #########################################################################
    def clicked_update_model(self):
        # @TODO
        return
