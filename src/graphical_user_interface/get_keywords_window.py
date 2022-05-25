# -*- coding: utf-8 -*-

"""
@author: L. Calvo-Bartolome
"""


# General imports
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget

# Local imports
from src.graphical_user_interface.messages import Messages
from src.graphical_user_interface.constants import Constants


class GetKeywordsWindow(QtWidgets.QDialog):
    """
    Class representing the window that is used for the attainment of a subcorpus
    from a given list of keywords, this list being selected by the user.
    """
    def __init__(self, tm):
        """
        Initializes a "GetKeywordsWindow" window.

        Parameters
        ----------
        tm : TaskManager 
            TaskManager object associated with the project
        """

        super(GetKeywordsWindow, self).__init__()

        # Load UI and configure default geometry of the window
        # #####################################################################
        uic.loadUi("UIs/get_labels_by_keywords.ui", self)
        self.init_ui()

        # ATTRIBUTES
        # #####################################################################
        self.tm = tm
        self.selectedKeywords = None
        self.selectedTag = None
        # Weight of the title. A word in the title is equivalent to wt repetitions
        # of the word in the description.
        self.wt_default = self.tm.global_parameters['keywords']['wt']
        self.wt = self.wt_default
        # Maximum number of elements in the output list
        self.n_max_default = self.tm.global_parameters['keywords']['n_max']
        self.n_max = self.n_max_default
        # Minimum score. Only docs scored strictly above s_min are selected
        self.s_min_default = self.tm.global_parameters['keywords']['s_min']
        self.s_min = self.s_min_default

        # Initialize parameters in the GUI
        self.init_params()

        # CONNECTION WITH HANDLER FUNCTIONS
        # #####################################################################
        self.get_labels_push_button.clicked.connect(
            self.clicked_select_keywords)

    def init_ui(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        self.setWindowIcon(QIcon('UIs/Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def center(self):
        """Centers the window at the middle of the screen at which the application is being executed.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_params(self):
        """Initializes the keywords parameters in the parameters' table within the GUI's "Get keywords" window,
        i.e. wt, n_max, and s_min. The default configuration of these parameters is read from the configuration file
        '/config/parameters.default.yaml'.
        """
        self.table_params.clearContents()
        self.table_params.setRowCount(1)
        self.table_params.setColumnCount(3)

        self.table_params.setItem(
            0, 0, QtWidgets.QTableWidgetItem(str(self.wt)))
        self.table_params.setItem(
            0, 1, QtWidgets.QTableWidgetItem(str(self.n_max)))
        self.table_params.setItem(
            0, 2, QtWidgets.QTableWidgetItem(str(self.s_min)))

    def update_params(self):
        """Updates the keywords parameters that are going to be used in the getting of the keywords based on the
        values read from the table within the GUI's "Get keywords" window that have been specified by the user.
        """
        if self.table_params.item(0, 0) is not None:
            self.wt = int(self.table_params.item(0, 0).text())
        else:
            self.wt = self.wt_default

        if self.table_params.item(0, 1) is not None:
            self.n_max = int(self.table_params.item(0, 1).text())
        else:
            self.n_max = self.n_max_default

        if self.table_params.item(0, 2) is not None:
            self.s_min = float(self.table_params.item(0, 2).text())
        else:
            self.s_min = self.s_min_default

        self.init_params()

    def show_suggested_keywords(self):
        """Displays the corresponding keywords based on the configuration parameters selected by the user on the top
        QTextEdit "text_edit_show_keywords".
        """
        suggested_keywords = self.tm.get_suggested_keywords()
        self.text_edit_show_keywords.setPlainText(suggested_keywords)

    def clicked_select_keywords(self):
        """Method to control the actions that are carried out at the time the "Select keywords" button of the "Get
        keywords window" is pressed by the user.
        """
        # Update configuration parameters to take into account the changes that the user could have introduced
        self.update_params()

        # Show warning message in case the user clicks on the "Select keywords" button without having previously
        # written the words that he wants to use as keywords
        if self.text_edit_get_keywords.toPlainText() == "":
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE, Messages.NO_KEYWORDS_SELECTED)
            return
        else:
            # Get keywords from the "text_edit_get_keywords" QTextEdit (positioned at the bottom left of the window)
            keywords = self.text_edit_get_keywords.toPlainText()
            # Split the keywords by commas, removing leading and trailing spaces
            _keywords = [x.strip() for x in keywords.split(',')]
            # Remove multiple spaces
            _keywords = [' '.join(x.split()) for x in _keywords]
            self.selectedKeywords = _keywords

        # Show warning message in case no tag for the file in which the subcorpus conformed based on the selected
        # keywords is going to be saved has been selected

        if self.line_edit_get_tag.text() == "":
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE, Messages.NO_TAG_SELECTED)
            return
        else:
            # Get selected tag
            self.selectedTag = str(self.line_edit_get_tag.text())

        # Hide window
        self.hide()
        # Clear QLineEdits and QTextEdits
        self.text_edit_show_keywords.setPlainText("")
        self.text_edit_get_keywords.setPlainText("")
        self.line_edit_get_tag.setText("")
        return
