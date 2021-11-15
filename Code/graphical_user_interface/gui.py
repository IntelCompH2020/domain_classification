# -*- coding: utf-8 -*-
"""
Class representing the main window of the application.

@author: lcalv
"""

import os
import time
import pathlib
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMenu, QButtonGroup
from PyQt5.QtCore import QThreadPool

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
        self.load_label_option = 0

        # INFORMATION BUTTONS
        ########################################################################
        self.infoButtonSelectCorpus.setIcon(QIcon('Images/help2.png'))
        self.infoButtonSelectCorpus.setToolTip(Messages.INFO_SELECT_CORPUS)
        self.infoButtonLoadLabels.setIcon(QIcon('Images/help2.png'))
        self.infoButtonLoadLabels.setToolTip(Messages.INFO_LOAD_LABELS)
        self.infoButtonKeywords.setIcon(QIcon('Images/help2.png'))
        self.infoButtonKeywords.setToolTip(Messages.INFO_TYPE_KEYWORDS)


        # CONFIGURE ELEMENTS IN THE "LOAD CORPUS VIEW"
        ########################################################################
        # self.configureTreeViewCorpusToLoad()

        self.treeViewSelectCorpus.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeViewSelectCorpus.customContextMenuRequested.connect(self.context_menu)
        self.treeViewSelectCorpus.doubleClicked.connect(self.clicked_select_corpus)
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
        self.pushButtonLoad.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage))
        self.pushButtonLoad.setIcon(QIcon('Images/settings.png'))

        self.load_labels_radio_buttons = QButtonGroup(self)
        self.load_labels_radio_buttons.addButton(self.loadLabelsOption1, 1)
        self.load_labels_radio_buttons.addButton(self.loadLabelsOption2, 2)
        self.load_labels_radio_buttons.addButton(self.loadLabelsOption3, 3)
        self.load_labels_radio_buttons.addButton(self.loadLabelsOption4, 4)
        self.load_labels_radio_buttons.buttonClicked.connect(self.clicked_load_labels_option)
        self.getSubcorpusPushButton.clicked.connect(self.clicked_load_subcorpus)

    ####################################################################################################################
    # LOAD CORPUS FUNCTIONS
    ####################################################################################################################
    def show_corpora(self):
        """Method to list all the corpora contained in the source folder selected by the user.
        """
        path = pathlib.Path(pathlib.Path(self.source_folder)).as_posix()
        print(self.source_folder)
        print(path)
        self.treeViewSelectCorpus.setModel(self.modelTreeView)
        self.treeViewSelectCorpus.setRootIndex(self.modelTreeView.index(path))
        # self.modelTreeView.setNameFilters(["*" + ".txt"])
        self.treeViewSelectCorpus.setSortingEnabled(True)
        return

    def context_menu(self):
        """Method control the opening of file in a text editor when an element
        from the dataset list in the mallet directory is clicked with the right
        mouse button.
        """
        menu = QtWidgets.QMenu()
        open = menu.addAction("Open file")
        open.triggered.connect(self.open_file)
        cursor = QtGui.QCursor()
        menu.exec_(cursor.pos())

    def open_file(self):
        """Method to open the content of a file in a text editor when called
        by context_menu.
        """
        index = self.treeViewSelectCorpus.currentIndex()
        file_path = self.modelTreeView.filePath(index)
        os.startfile(file_path)

    def execute_load_corpus(self):
        """Method to control the execution of the training of a model.
        """
        self.tm.load_corpus(self.corpus_selected_name)
        return "Done."

    def do_after_load_corpus(self):
        # Showing messages in the status bar, pop up window, and corpus label
        self.statusBar().showMessage("'" + self.corpus_selected_name + "' was selected as corpus.", 10000)
        QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE,
                                          "The corpus '" + self.corpus_selected_name + "' has been loaded.")

        self.labelCorpusSelectedIs.setText(str(self.corpus_selected_path))

    def clicked_select_corpus(self):
        """Method to control the selection of a new corpus by double clicking into one of the items of the corpus list
        within the selected source folder, as well as its loading as dataframe into the TaskManager object.
        """
        index = self.treeViewSelectCorpus.currentIndex()
        self.corpus_selected_path = self.modelTreeView.filePath(index)
        self.corpus_selected_name = self.modelTreeView.fileName(index)

        # Loading corpus into the TaskManager object as dataframe
        self.statusBar().showMessage("'The corpus " + self.corpus_selected_name + "' is being loaded.", 30000)
        execute_in_thread(self, self.execute_load_corpus, self.do_after_load_corpus, False)

    ####################################################################################################################
    # LOAD LABELS FUNCTIONS
    ####################################################################################################################
    def clicked_load_labels_option(self):
        if self.tm.corpus_name is None:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.INCORRECT_INPUT_PARAM_SELECTION)
        else:
            if self.load_labels_radio_buttons.checkedId() == 1:
                print("Import labels from a source file")
                self.load_label_option = 1
            elif self.load_labels_radio_buttons.checkedId() == 2:
                print("Get subcorpus from a given list of keywords")
                self.load_label_option = 2
                suggested_keywords = self.tm.get_suggested_keywords()
                self.textEditShowKeywords.setPlainText(suggested_keywords)
            elif self.load_labels_radio_buttons.checkedId() == 3:
                print("Get subcorpus from a topic selection function")
                self.load_label_option = 3
            else:
                print("Get subcorpus from documents defining categories")
                self.load_label_option = 4
        return

    def clicked_load_subcorpus(self):
        if self.load_label_option == 0:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.INCORRECT_NO_LABEL_OPTION_SELECTED)
        else:
            if self.load_label_option == 1:
                self.tm.import_labels()
            elif self.load_label_option == 2:
                keywords = self.lineEditGetKeywords.text()
                # Split by commas, removing leading and trailing spaces
                _keywords = [x.strip() for x in keywords.split(',')]
                # Remove multiple spaces
                _keywords = [' '.join(x.split()) for x in _keywords]
                self.tm.get_labels_by_keywords_gui(_keywords)
            elif self.load_label_option == 3:
                self.tm.get_labels_by_topics()
            else:
                self.tm.get_labels_by_definitions()

        QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE,
                                          "Labels have been loaded.")

        # Reset after loading labels
        self.load_labels_radio_buttons.setExclusive(False)
        self.loadLabelsOption1.setChecked(False)
        self.loadLabelsOption2.setChecked(False)
        self.loadLabelsOption3.setChecked(False)
        self.loadLabelsOption4.setChecked(False)
        self.load_labels_radio_buttons.setExclusive(True)
        self.load_label_option == 0
        self.textEditShowKeywords.setPlainText("")
        self.lineEditGetKeywords.setText("")
        return
