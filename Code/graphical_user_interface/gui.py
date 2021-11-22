# -*- coding: utf-8 -*-
"""
Class representing the main window of the application.

@author: lcalv
"""

import os
import pathlib
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QButtonGroup
from PyQt5.QtCore import QThreadPool
import time

from Code.graphical_user_interface.AnalyzeKeywordsWindow import AnalyzeKeywordsWindow
from Code.graphical_user_interface.GetKeywordsWindow import GetKeywordsWindow
from Code.graphical_user_interface.Messages import Messages
from Code.graphical_user_interface.util import toggleMenu, execute_in_thread


class GUI(QtWidgets.QMainWindow):
    def __init__(self, projectFolder, sourceFolder, tm):
        super(GUI, self).__init__()

        # Load UI and configure default geometry of the window
        ########################################################################
        uic.loadUi("UIS/DomainClassifier.ui", self)

        self.setGeometry(100, 60, 2000, 1600)
        self.centralwidget.setGeometry(100, 60, 2000, 1600)
        self.animation = QtCore.QPropertyAnimation(self.frame_left_menu, b"minimumWidth")

        # ATTRIBUTES
        #######################################################################
        self.source_folder = sourceFolder
        self.project_folder = projectFolder
        self.tm = tm
        self.corpus_selected_path = None
        self.corpus_selected_name = ""
        self.get_label_option = 0
        self.get_keywords_window = GetKeywordsWindow(tm)
        self.analyze_keywords_window = AnalyzeKeywordsWindow(tm)

        # INFORMATION BUTTONS
        ########################################################################
        self.info_button_select_corpus.setIcon(QIcon('Images/help2.png'))
        self.info_button_select_corpus.setToolTip(Messages.INFO_SELECT_CORPUS)
        self.info_button_get_labels.setIcon(QIcon('Images/help2.png'))
        self.info_button_get_labels.setToolTip(Messages.INFO_GET_LABELS)
        self.info_button_load_reset_labels.setIcon(QIcon('Images/help2.png'))
        self.info_button_load_reset_labels.setToolTip(Messages.INFO_LOAD_RESET_LABELS)

        # CONFIGURE ELEMENTS IN THE "LOAD CORPUS VIEW"
        ########################################################################
        # self.configureTreeViewCorpusToLoad()

        self.tree_view_select_corpus.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_view_select_corpus.customContextMenuRequested.connect(self.context_menu)
        self.tree_view_select_corpus.doubleClicked.connect(self.clicked_select_corpus)
        self.modelTreeView = QtWidgets.QFileSystemModel(nameFilterDisables=False)
        self.modelTreeView.setRootPath((QtCore.QDir.rootPath()))
        self.show_corpora()

        # THREADS FOR EXECUTING PARALLEL TRAIN MODELS, TRAIN SUBMODELS, AND PYLDAVIS
        ########################################################################
        self.thread_pool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.thread_pool.maxThreadCount())
        self.worker = None
        self.loading_window = None

        # TOGGLE MENU
        ########################################################################
        self.toggleButton.clicked.connect(lambda: toggleMenu(self, 250))
        self.toggleButton.setIcon(QIcon('Images/menu.png'))

        # PAGES
        ########################################################################
        # PAGE 1: Load corpus/ labels
        self.pushButtonLoad.clicked.connect(lambda: self.tabs.setCurrentWidget(self.page_load))
        self.pushButtonLoad.setIcon(QIcon('Images/settings.png'))

        self.get_labels_radio_buttons = QButtonGroup(self)
        self.get_labels_radio_buttons.addButton(self.get_labels_option1, 1)
        self.get_labels_radio_buttons.addButton(self.get_labels_option2, 2)
        self.get_labels_radio_buttons.addButton(self.get_labels_option3, 3)
        self.get_labels_radio_buttons.addButton(self.get_labels_option4, 4)
        self.get_labels_radio_buttons.addButton(self.get_labels_option5, 5)
        self.get_labels_radio_buttons.buttonClicked.connect(self.clicked_get_labels_option)
        self.get_labels_push_button.clicked.connect(self.clicked_get_labels)

    ####################################################################################################################
    # LOAD CORPUS FUNCTIONS
    ####################################################################################################################
    def show_corpora(self):
        """Method to list all the corpora contained in the source folder selected by the user.
        """
        path = pathlib.Path(pathlib.Path(self.source_folder)).as_posix()
        print(self.source_folder)
        print(path)
        self.tree_view_select_corpus.setModel(self.modelTreeView)
        self.tree_view_select_corpus.setRootIndex(self.modelTreeView.index(path))
        # self.modelTreeView.setNameFilters(["*" + ".txt"])
        self.tree_view_select_corpus.setSortingEnabled(True)
        return

    def context_menu(self):
        """
        Method for controlling the opening of a file when an element from the source data folder is right clicked.
        """
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open file")
        open.triggered.connect(self.open_file)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def open_file(self):
        """
        Method to open the content of a file in a text editor when called by the context_menu() function.
        """
        index = self.tree_view_select_corpus.currentIndex()
        file_path = self.modelTreeView.filePath(index)
        os.startfile(file_path)

    def execute_load_corpus(self):
        """
        Method to control the execution of the loading of a corpus on a secondary thread while the GUI execution
        is maintained in the main thread.
        """
        self.tm.load_corpus(self.corpus_selected_name)
        return "Done."

    def do_after_load_corpus(self):
        """
        Method to be executed after the loading of the corpus has been completed.
        """
        # Showing messages in the status bar, pop up window, and corpus label
        self.statusBar().showMessage("'" + self.corpus_selected_name + "' was selected as corpus.", 10000)
        QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE,
                                          "The corpus '" + self.corpus_selected_name + "' has been loaded in the "
                                                                                       "current session.")

        self.label_corpus_selected_is.setText(str(self.corpus_selected_path))

    def clicked_select_corpus(self):
        """
        Method to control the selection of a new corpus by double clicking into one of the items of the corpus list
        within the selected source folder, as well as its loading as dataframe into the TaskManager object.
        """
        index = self.tree_view_select_corpus.currentIndex()
        self.corpus_selected_path = self.modelTreeView.filePath(index)
        self.corpus_selected_name = self.modelTreeView.fileName(index)

        # Loading corpus into the TaskManager object as dataframe
        self.statusBar().showMessage("'The corpus " + self.corpus_selected_name + "' is being loaded.", 30000)
        execute_in_thread(self, self.execute_load_corpus, self.do_after_load_corpus, False)

    ####################################################################################################################
    # LOAD LABELS FUNCTIONS
    ####################################################################################################################
    def clicked_get_labels_option(self):
        """
        Method to control the functionality associated with the selection of each of the QRadioButtons associated with
        the labels' getting.
        Only one QRadioButton can be selected at a time.
        """
        if self.tm.corpus_name is None:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.INCORRECT_INPUT_PARAM_SELECTION)
        else:
            if self.get_labels_radio_buttons.checkedId() == 1:
                print("Import labels from a source file")
                self.get_label_option = 1
            elif self.get_labels_radio_buttons.checkedId() == 2:
                print("Get subcorpus from a given list of keywords")
                self.get_label_option = 2
                # Show the window for selecting the keywords
                self.get_keywords_window.show_suggested_keywords()
                self.get_keywords_window.show()
            elif self.get_labels_radio_buttons.checkedId() == 3:
                print("Analyze the presence of selected keywords in the corpus")
                self.get_label_option = 3
                # Show the window for selecting the keywords in case they have not been selected yet
                if self.tm.keywords is None:
                    QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE, Messages.INFO_NO_ACTIVE_KEYWORDS)
                    # Show the window for selecting the keywords
                    self.get_keywords_window.show_suggested_keywords()
                    self.get_keywords_window.exec()
                else:
                    QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE, Messages.INFO_NO_ACTIVE_KEYWORDS)
                # Show the window for the analysis of the keywords
                self.analyze_keywords_window.do_analysis()
                self.analyze_keywords_window.exec()

            elif self.get_labels_radio_buttons.checkedId() == 4:
                print("Get subcorpus from a topic selection function")
                self.get_label_option = 4
            else:
                print("Get subcorpus from documents defining categories")
                self.get_label_option = 5
        return

    def clicked_get_labels(self):
        """
        Method for performing the getting of the labels according to the method selected for it.
        """
        if self.get_label_option == 0:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.INCORRECT_NO_LABEL_OPTION_SELECTED)
        else:
            if self.get_label_option == 1:
                self.tm.import_labels()
            elif self.get_label_option == 2 or self.get_label_option == 3:
                self.tm.get_labels_by_keywords_gui(self.get_keywords_window.selectedKeywords,
                                                   self.get_keywords_window.selectedTag)
            elif self.get_label_option == 4:
                self.tm.get_labels_by_topics()
            else:
                self.tm.get_labels_by_definitions()

        QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE,
                                          "Labels have been loaded.")

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
