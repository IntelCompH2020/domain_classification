# -*- coding: utf-8 -*-
"""
Class representing the main window of the application.

@author: lcalv
"""

import os
import pathlib
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon

from Code.graphical_user_interface.Messages import Messages
from Code.graphical_user_interface.util import toggleMenu


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

        # INFORMATION BUTTONS
        ########################################################################
        self.infoButtonSelectCorpus.setIcon(QIcon('Images/help2.png'))
        self.infoButtonSelectCorpus.setToolTip(Messages.INFO_SELECT_CORPUS)

        # CONFIGURE ELEMENTS IN THE "LOAD CORPUS VIEW"
        ########################################################################
        # self.configureTreeViewCorpusToLoad()

        self.treeViewSelectCorpus.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.treeViewSelectCorpus.customContextMenuRequested.connect(self.context_menu)
        self.treeViewSelectCorpus.doubleClicked.connect(self.clicked_select_corpus)
        self.modelTreeView = QtWidgets.QFileSystemModel(nameFilterDisables=False)
        self.modelTreeView.setRootPath((QtCore.QDir.rootPath()))
        self.show_corpora()

        # TOGGLE MENU
        ########################################################################
        self.toggleButton.clicked.connect(lambda: toggleMenu(self, 250))
        self.toggleButton.setIcon(QIcon('Images/menu.png'))

        # PAGES
        ########################################################################
        # PAGE 1: Load corpus
        self.pushButtonLoad.clicked.connect(lambda: self.tabs.setCurrentWidget(self.tabsPage))
        self.pushButtonLoad.setIcon(QIcon('Images/settings.png'))

    def clicked_select_corpus(self):
        """Method to control the selection of a new corpus by double clicking into one of the items of the corpus list
        within the selected source folder, as well as its loading as dataframe into the TaskManager object.
        """
        index = self.treeViewSelectCorpus.currentIndex()
        corpus_selected_path = self.modelTreeView.filePath(index)
        corpus_selected_name = self.modelTreeView.fileName(index)

        # Loading corpus into the TaskManager object as dataframe
        self.tm.load_corpus(corpus_selected_name)

        # Showing messages in the status bar, pop up window, and corpus label
        self.statusBar().showMessage("'" + corpus_selected_name + "' was selected as corpus.", 10000)
        QtWidgets.QMessageBox.information(self, Messages.DC_MESSAGE,
                                          "'The corpus " + corpus_selected_name + "' has been loaded.")
        self.labelCorpusSelectedIs.setText(str(corpus_selected_path))

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