# -*- coding: utf-8 -*-
"""
Main class for the Domain Classification GUI. It starts the GUI application by first creating an starting window,
which after the user has selected the required input parameters (i.e. project and source folder), redirects the
user to the main window of the application.

@author: lcalv
"""

import sys
import argparse
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from pathlib import Path

from Code.graphical_user_interface.MainWindow import *
from Code.graphical_user_interface.Messages import Messages
from Code.task_manager import TaskManager


class PreConfig(QDialog):
    def __init__(self):
        super(PreConfig, self).__init__()
        # Load UI
        loadUi("UIs/menuConfig.ui", self)

        # Get home in any operating system
        self.home = str(Path.home())

        # Variables for saving the project and source data folders
        self.projectFolder = ""
        self.sourceFolder = ""

        # Update image
        pixmap = QPixmap('Images/dc_logo.png')
        self.label.setPixmap(pixmap)

        # Update project and source folder in the GUI if provided through the command line
        self.showProjectFolder.setText(args.p) if args.p is not None else print("p not provided")
        self.showSourceDataFolder.setText(args.source) if args.source is not None else print("source not provided")

        self.selectProjectFolder.clicked.connect(self.getProjectFolder)
        self.selectSourceDataFolder.clicked.connect(self.getSourceDataFolder)
        self.start.clicked.connect(self.startApplication)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def getProjectFolder(self):
        self.projectFolder = QFileDialog.getExistingDirectory(self, 'Create or select an an existing project',
                                                              self.home)
        self.showProjectFolder.setText(self.projectFolder)

    def getSourceDataFolder(self):
        self.sourceFolder = QFileDialog.getExistingDirectory(self, 'Select the source data folder', self.home)
        self.showSourceDataFolder.setText(self.sourceFolder)

    def startApplication(self):
        # We save the project and source path if provided through the command line
        self.projectFolder = args.p if args.p is not None else ""
        self.sourceFolder = args.source if args.source is not None else ""

        # We show a warning message if one or  both folders have not selected
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
        # widget.resize(2480, 1360)
        return


# Handle high resolution displays:
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, False)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, False)

########################################################################
# Main
########################################################################

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to a new or an existing project")
parser.add_argument('--source', type=str, default=None,
                    help="path to the source data folder")
args = parser.parse_args()

# Configure font and style set
default_font = QtGui.QFont('Arial', 10)
default_font.setPixelSize(25)  # 20 for Ubuntu 20.04.3; 25 for Windows
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
height = 2480  # 1540
weight = 1360  # 880
widget.resize(height, weight)
widget.show()
sys.exit(app.exec_())


