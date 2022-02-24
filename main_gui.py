# -*- coding: utf-8 -*-
"""
@author: lcalv
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from pathlib import Path

from src.graphical_user_interface.main_window import *
from src.graphical_user_interface.messages import Messages
from src.task_manager import TaskManagerGUI
from src.graphical_user_interface.output_wrapper import OutputWrapper


class PreConfig(QDialog):
    """
    Main class for the Domain Classification GUI. It starts the GUI application by
    first creating a starting window, which after a correct selection of the input
    parameters (i.e. project and source folder), redirects the user to the main
    window of the application.

    It can be invoked in the following ways:

    1) python main_gui.py --p path_to_project_folder --source path_to_source_folder
        ---> The project and source folder are automatically updated in the
             starting window, and the START button can be directly clicked, without
             the necessity of further configurations.
    2) python main_gui.py
        ---> The project and source folder need to be manually selected by clicking
             on their respective buttons.
    """

    def __init__(self, widget, args):
        super(PreConfig, self).__init__()
        """
        Initializes the main class for the Domain Classification GUI.

        Parameters
        ----------
        widget : QtWidgets.QStackedWidget
            Window to which the application's main window is attached to
        args : argparse.Namespace
            List of positional arguments leftover after parsing options.
        """

        # Load UI
        loadUi("UIs/menuConfig.ui", self)
        self.center()

        self.widget = widget
        self.args = args

        # Redirect stdout and stderr
        self.stdout = OutputWrapper(self, True)
        self.stderr = OutputWrapper(self, False)

        # Get home in any operating system
        self.home = str(Path.home())

        # Variables for saving the project and source data folders
        self.sourceFolder = self.args.source if self.args.source is not None else ""
        self.projectFolder = self.args.p if self.args.p is not None else ""

        # Update image
        pixmap = QPixmap('Images/dc_logo.png')
        self.label.setPixmap(pixmap)

        # Update project and source folder in the GUI if provided through the
        # command line
        self.showProjectFolder.setText(
            self.args.p) if self.args.p is not None else print("p not provided")
        self.showSourceDataFolder.setText(
            self.args.source) if self.args.source is not None else print(
            "source not provided")

        # CONNECTION WITH HANDLER FUNCTIONS
        #######################################################################
        self.selectProjectFolder.clicked.connect(self.get_project_folder)
        self.selectSourceDataFolder.clicked.connect(self.get_source_data_folder)
        self.start.clicked.connect(self.start_application)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def get_project_folder(self):
        print(self.args.p)
        self.projectFolder = \
            QFileDialog.getExistingDirectory(
                self, 'Create or select an an existing project', self.home)
        self.showProjectFolder.setText(self.projectFolder)

    def get_source_data_folder(self):
        self.sourceFolder = \
            QFileDialog.getExistingDirectory(
                self, 'Select the source data folder', self.home)
        self.showSourceDataFolder.setText(self.sourceFolder)

    def start_application(self):
        # We show a warning message if one or  both folders have not selected
        if self.projectFolder == "" or self.sourceFolder == "":
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE,
                Messages.INCORRECT_INPUT_PARAM_SELECTION)
            return

        # Create the TaskManager object
        tm = TaskManagerGUI(self.projectFolder, path2source=self.sourceFolder)
        if len(os.listdir(self.projectFolder)) == 0:
            print("A new project folder was selected. Proceeding with "
                  "its configuration...")
            tm.create()
            tm.setup()
        else:
            print("An existing project folder was selected. Proceeding with "
                  "its loading...")
            tm.load()

        # Change to the main menu
        main_window = MainWindow(self.projectFolder, self.sourceFolder, tm, self.widget, self.stdout, self.stderr)
        self.widget.addWidget(main_window)
        self.widget.setCurrentIndex(self.widget.currentIndex() + 1)
        self.widget.showMaximized()
        return


########################################################################
# Main
########################################################################
def main():
    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default="/Users/lbartolome/Documents/Intelcomp/project_folder",
                        help="path to a new or an existing project")
    parser.add_argument('--source', type=str, default="/Users/lbartolome/Documents/Intelcomp/datasets",
                        help="path to the source data folder")
    args = parser.parse_args()

    # Create application
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('Images/dc_logo.png'))

    # Configure widgets
    widget = QtWidgets.QStackedWidget()
    widget.setWindowTitle(Messages.WINDOW_TITLE)

    # Create main menu window
    config_window = PreConfig(widget, args)
    widget.addWidget(config_window)
    # height = 2480  # 1540
    # weight = 1360  # 880
    # widget.resize(height, weight)
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
