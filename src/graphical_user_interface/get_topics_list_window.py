# -*- coding: utf-8 -*-
"""
Class representing the window for controlling the getting of a subcorpus from
a given list of topics

@author: lcalv
"""
import numpy as np
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget

from src.graphical_user_interface.messages import Messages


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
        self.n_max_default = 2000
        self.n_max = self.n_max_default
        # Significance threshold.
        self.s_min_default = 0.2
        self.s_min = self.s_min_default

        self.init_params()

        # INFORMATION BUTTONS
        #######################################################################
        self.info_button_topic_list.setIcon(QIcon('Images/help2.png'))
        self.info_button_topic_list.setToolTip(Messages.INFO_TOPIC_LIST)
        self.info_button_topic_list.setIconSize(
            0.75 * QSize(self.info_button_topic_list.width(),
                         self.info_button_topic_list.height()))
        self.info_button_selected_tag.setIcon(QIcon('Images/help2.png'))
        self.info_button_selected_tag.setIconSize(
            0.75 * QSize(self.info_button_selected_tag.width(),
                         self.info_button_selected_tag.height()))
        self.info_button_selected_tag.setToolTip(Messages.INFO_TAG)

        # TABLE TOOL TIPS
        #######################################################################
        self.table_widget_topics_weight.setToolTip(Messages.INFO_TABLE_WEIGHTS)
        self.table_params.setToolTip(Messages.INFO_TABLE_PARAMETERS_TOPICS)

        # CONNECTION WITH HANDLER FUNCTIONS
        #######################################################################
        self.table_widget_topics_weight.cellChanged.connect(
            self.updated_topic_weighted_list)
        self.get_topic_list_push_button.clicked.connect(
            self.clicked_get_topic_list)
        self.table_params.cellChanged.connect(self.update_params)

    def initUI(self):
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_params(self):
        self.table_params.clearContents()
        self.table_params.setRowCount(1)
        self.table_params.setColumnCount(2)

        print(self.n_max)
        print(self.s_min)
        self.table_params.setItem(
            0, 0, QtWidgets.QTableWidgetItem(str(self.n_max)))
        self.table_params.setItem(
            0, 1, QtWidgets.QTableWidgetItem(str(self.s_min)))

    def update_params(self):
        if self.table_params.item(0, 0) is not None:
            self.n_max = self.table_params.item(0, 0).text()
        else:
            self.n_max = self.n_max_default

        if self.table_params.item(0, 1) is not None:
            self.s_min = self.table_params.item(0, 1).text()
        else:
            self.s_min = self.s_min_default

        self.init_params()

    def show_topics(self):
        self.tw, self.T, self.df_metadata = self.tm.get_topic_words(
            self.n_max, self.s_min)
        n_topics = len(self.tw)

        # TABLE FOR SHOWING TOPIC ID AND CHEMICAL DESCRIPTION
        #####################
        # column 0 = topic nr
        # column 1 = weight - > the user introduces it
        # column 2 = topic description
        #####################
        self.table_widget_topic_list.clearContents()
        self.table_widget_topic_list.setRowCount(n_topics)
        self.table_widget_topic_list.setColumnCount(2)

        for i in range(n_topics):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.table_widget_topic_list.setItem(i, 0, item_topic_nr)
            item_topic_description = QtWidgets.QTableWidgetItem(
                str(self.tw[i]))
            self.table_widget_topic_list.setItem(i, 1, item_topic_description)

        # TABLE IN WHICH THE USER INTRODUCES THE WEIGHTS FOR THE TOPICS
        #####################
        # column 0 = weight introduced by the user
        #####################
        self.table_widget_topics_weight.clearContents()
        self.table_widget_topics_weight.setRowCount(n_topics)
        self.table_widget_topics_weight.setColumnCount(1)
        return

    def updated_topic_weighted_list(self):
        # Get names from the right table and show in the left one
        topic_weighted_list = ""
        for i in np.arange(0, self.table_widget_topics_weight.rowCount(), 1):
            if self.table_widget_topics_weight.item(i, 0) is not None:
                weight = self.table_widget_topics_weight.item(i, 0).text()
                topic_weighted_list += str(i) + "," + weight + ","
        print(topic_weighted_list)
        topic_weighted_list = topic_weighted_list[:-1]
        print(topic_weighted_list)
        self.line_topic_list.setText(topic_weighted_list)

    def clicked_get_topic_list(self):
        if self.line_topic_list.text() is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE, Messages.NO_TOPIC_LIST_SELECTED)
        else:
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
        if self.line_edit_get_tag.text() is None:
            QtWidgets.QMessageBox.warning(
                self, Messages.DC_MESSAGE, Messages.NO_TAG_SELECTED)
        else:
            self.selectedTag = str(self.line_edit_get_tag.text())

        self.hide()
        self.line_topic_list.setText("")
        self.line_edit_get_tag.setText("")
        return
