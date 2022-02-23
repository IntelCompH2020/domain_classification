# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                     CLASS TOPIC LISTS WINDOW                           ***
******************************************************************************
Class representing the window in charge of getting a subcorpus from a given
list of topics, such a list being specified by the user.
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
# General imports
import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget

# Local imports
from src.graphical_user_interface.messages import Messages
from src.graphical_user_interface.constants import Constants


class GetTopicsListWindow(QtWidgets.QDialog):
    def __init__(self, tm):
        super(GetTopicsListWindow, self).__init__()

        # Load UI and configure default geometry of the window
        #######################################################################
        uic.loadUi("UIS/get_labels_by_topics.ui", self)
        self.initUI()

        # ATTRIBUTES
        #######################################################################
        self.tm = tm
        self.selectedTag = None
        self.tw = None
        self.T = None
        self.df_metadata = None
        # Maximum number of elements in the output list
        self.n_max_default = self.tm.global_parameters['topics']['n_max']
        self.n_max = self.n_max_default
        # Significance threshold.
        self.s_min_default = self.tm.global_parameters['topics']['s_min']
        self.s_min = self.s_min_default

        # Initialize parameters in the GUI
        self.init_params()

        # INFORMATION BUTTONS
        #######################################################################
        self.info_button_topic_list.setIcon(QIcon('Images/help2.png'))
        self.info_button_topic_list.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_topic_list.width(),
                                            self.info_button_topic_list.height()))
        self.info_button_selected_tag.setIcon(QIcon('Images/help2.png'))
        self.info_button_selected_tag.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_selected_tag.width(),
                                            self.info_button_selected_tag.height()))
        self.info_button_introduce_weights.setIcon(QIcon('Images/help2.png'))
        self.info_button_introduce_weights.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_selected_tag.width(),
                                            self.info_button_selected_tag.height()))
        self.info_button_get_labels_by_topics_parameters.setIcon(QIcon('Images/help2.png'))
        self.info_button_get_labels_by_topics_parameters.setIconSize(
            Constants.BUTTONS_SCALE * QSize(self.info_button_selected_tag.width(),
                                            self.info_button_selected_tag.height()))

        # CONNECTION WITH HANDLER FUNCTIONS
        #######################################################################
        self.table_widget_topics_weight.cellChanged.connect(
            self.updated_topic_weighted_list)
        self.get_topic_list_push_button.clicked.connect(
            self.clicked_get_topic_list)

    def initUI(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application,
        the application's title, and the position of the window at its opening.
        """
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
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
        """Initializes the topics parameters in the parameters' table within the GUI's "Get topic list" window,
        i.e. n_max and s_min. The default configuration of these parameters is read from the configuration file
        '/config/parameters.default.yaml'.
        """
        self.table_params.clearContents()
        self.table_params.setRowCount(1)
        self.table_params.setColumnCount(2)

        self.table_params.setItem(
            0, 0, QtWidgets.QTableWidgetItem(str(self.n_max)))
        self.table_params.setItem(
            0, 1, QtWidgets.QTableWidgetItem(str(self.s_min)))

    def update_params(self):
        """Updates the topics parameters that are going to be used in the getting of the keywords based on the values
        read from the table within the GUI's "Get topic list" window that have been specified by the user.
        """
        if self.table_params.item(0, 0) is not None:
            self.n_max = int(self.table_params.item(0, 0).text())
        else:
            self.n_max = self.n_max_default

        if self.table_params.item(0, 1) is not None:
            self.s_min = float(self.table_params.item(0, 1).text())
        else:
            self.s_min = self.s_min_default

        self.init_params()

    def show_topics(self):
        """Configures the "table_widget_topic_list" and "table_widget_topics_weight" tables to have the appropriate
        number of columns and rows based on the available topics, and fills out the "table_widget_topic_list" table
        with the id and corresponding chemical description of each of the topics.
        """
        # Get topic words       
        self.tw, self.T, self.df_metadata = self.tm.get_topic_words(
            self.n_max, self.s_min)
        n_topics = len(self.tw)

        # Configure "table_widget_topic_list" table (positioned at the top right)
        # TABLE FOR SHOWING TOPIC ID AND CHEMICAL DESCRIPTION
        ######################################################
        # column 0 = topic nr
        # column 1 = weight - > the user introduces it
        # column 2 = topic description
        ######################################################
        self.table_widget_topic_list.clearContents()
        self.table_widget_topic_list.setRowCount(n_topics)
        self.table_widget_topic_list.setColumnCount(2)

        # Fill out "table_widget_topic_list" table
        for i in range(n_topics):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.table_widget_topic_list.setItem(i, 0, item_topic_nr)
            item_topic_description = QtWidgets.QTableWidgetItem(
                str(self.tw[i]))
            self.table_widget_topic_list.setItem(i, 1, item_topic_description)

        # Configure "table_widget_topics_weight" table (positioned at the top left)
        # TABLE IN WHICH THE USER INTRODUCES THE WEIGHTS FOR THE TOPICS
        #################################################################
        # column 0 = weight introduced by the user
        #################################################################
        self.table_widget_topics_weight.clearContents()
        self.table_widget_topics_weight.setRowCount(n_topics)
        self.table_widget_topics_weight.setColumnCount(1)
        return

    def updated_topic_weighted_list(self):
        """Generates the topic weighted list based on the weights that the user has introduced on the "table_widget_topics_weight" table
        """
        # Get weights from the left table
        topic_weighted_list = ""
        for i in np.arange(0, self.table_widget_topics_weight.rowCount(), 1):
            if self.table_widget_topics_weight.item(i, 0) is not None:
                weight = self.table_widget_topics_weight.item(i, 0).text()
                topic_weighted_list += str(i) + "," + weight + ","
        # Remove empty space 
        topic_weighted_list = topic_weighted_list[:-1]
        self.line_topic_list.setText(topic_weighted_list)

    def clicked_get_topic_list(self):
        """Method to control the actions that are carried out at the time the "Select weighted topic list" button of
        the "Get topics window" is pressed by the user.
        """
        # Update configuration parameters to take into account the changes that the user could have
        # introduced
        self.update_params()

        # Show warning message in case the user clicks on the "Select weighted topic lists" button without having
        # previously written the weights that the user wants to use for the generation of the weighted topic list
        if self.line_topic_list.text() is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE, Messages.NO_TOPIC_LIST_SELECTED)
        else:
            # Get topic list
            topic_list = str(self.line_topic_list.text())
            tw_list = topic_list.split(',')

            # Get topic indices as integers
            keys = [int(k) for k in tw_list[::2]]
            # Get topic weights as floats
            weights = [float(w) for w in tw_list[1::2]]

            # Normalize weights
            sum_w = sum(weights)
            weights = [w / sum_w for w in weights]

            # Store in dictionary
            self.tw = dict(zip(keys, weights))

        # Show warning message in case no tag for the file in which the subcorpus conformed based on the selected
        # topics is going to be saved has been selected
        if self.line_edit_get_tag.text() is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE, Messages.NO_TAG_SELECTED)
        else:
            # Get selected tag
            self.selectedTag = str(self.line_edit_get_tag.text())

        # Hide window
        self.hide()
        # Clear QLineEdits
        self.line_topic_list.setText("")
        self.line_edit_get_tag.setText("")
        return
