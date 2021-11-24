# -*- coding: utf-8 -*-
"""
Main class for the Domain Classification GUI. It starts the GUI application by first creating an starting window,
which after the user has selected the required input parameters (i.e. project and source folder), redirects the
user to the main window of the application.

@author: lcalv
"""

import sys
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from pathlib import Path
from PyQt5.QtGui import QPixmap

from Code.graphical_user_interface.MainWindow import *
from Code.graphical_user_interface.Messages import Messages
from Code.task_manager import TaskManager


class PreConfig(QDialog):
    def __init__(self):
        super(PreConfig, self).__init__()
        loadUi("UIs/menuConfig.ui", self)
        self.setWindowTitle(Messages.WINDOW_TITLE)
        # font = QtGui.QFont('Arial')
        # font.setStyleHint(QtGui.QFont.TypeWriter)
        # font.setPixelSize(10)
        # self.setFont(font)

        # Get home in any operating system
        self.home = str(Path.home())

        # Variables for saving the project and source data folders
        self.projectFolder = ""
        self.sourceFolder = ""

        # Update image
        pixmap = QPixmap('Images/dc_logo.png')
        self.label.setPixmap(pixmap)

        self.selectProjectFolder.clicked.connect(self.getProjectFolder)
        self.selectSourceDataFolder.clicked.connect(self.getSourceDataFolder)
        self.start.clicked.connect(self.startApplication)

    def getProjectFolder(self):
        self.projectFolder = QFileDialog.getExistingDirectory(self, 'Create or select an an existing project',
                                                              self.home)
        self.showProjectFolder.setText(self.projectFolder)

    def getSourceDataFolder(self):
        self.sourceFolder = QFileDialog.getExistingDirectory(self, 'Select the source data folder', self.home)
        self.showSourceDataFolder.setText(self.sourceFolder)

    def startApplication(self):
        # We show a warning message if both folders have not selected
        if self.projectFolder == "" or self.sourceFolder == "":
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.INCORRECT_INPUT_PARAM_SELECTION)
            return

        # We create the TaskManager object
        tm = TaskManager(self.projectFolder, path2source=self.sourceFolder)
        if len(os.listdir(self.projectFolder)) == 0:
            print("A new project folder was selected. Proceeding with its configuration...")
            # tm.create()
            tm.configure_project_folder()
            tm.setup()
        else:
            print("An existing project folder was selected. Proceeding with its loading...")
            tm.load()

        # We change to the main menu
        mainWindow = MainWindow(self.projectFolder, self.sourceFolder, tm)
        widget.addWidget(mainWindow)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        widget.resize(2480, 1360)
        return


# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

########################################################################
# Main
########################################################################
# Configure font and style set
default_font = QtGui.QFont('Arial', 10)
default_font.setPixelSize(25)
QtWidgets.QApplication.setStyle("fusion")
QtWidgets.QApplication.setFont(default_font)
# Create application
app = QApplication(sys.argv)
app.setWindowIcon(QIcon('Images/dc_logo.png'))
app.setFont(default_font)
widget = QtWidgets.QStackedWidget()
widget.setWindowTitle(Messages.WINDOW_TITLE)
width = widget.frameGeometry().width()
height = widget.frameGeometry().height()
print(height)
print(width)
# Create main menu window
configWindow = PreConfig()
widget.addWidget(configWindow)
widget.resize(1540, 880)
widget.show()
sys.exit(app.exec_())


