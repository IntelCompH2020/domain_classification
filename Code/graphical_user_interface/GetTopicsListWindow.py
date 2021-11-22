# -*- coding: utf-8 -*-
"""
Class representing the window for controlling the getting of a subcorpus from a given list of keywords

@author: lcalv
"""

from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtGui import QIcon

from Code.graphical_user_interface.Messages import Messages


class GetTopicsListWindow(QtWidgets.QDialog):
    def __init__(self, tm):
        super(GetTopicsListWindow, self).__init__()

        # Load UI and configure default geometry of the window
        ########################################################################
        uic.loadUi("UIS/get_labels_by_topics.ui", self)

        self.setGeometry(100, 60, 2000, 1600)

        # ATTRIBUTES
        self.tm = tm
        self.selectedKeywords = None
        self.selectedTag = None

        # INFORMATION BUTTONS
        ########################################################################
        self.info_button_topic_list.setIcon(QIcon('Images/help2.png'))
        self.info_button_topic_list.setToolTip(Messages.INFO_TOPIC_LIST)

        self.get_topic_list_push_button.clicked.connect(self.clicked_get_topic_list)

    def show_topics(self):
        topic_words = self.tm.get_topic_words_gui()
        n_topics = len(topic_words)
        #####################
        # column 0 = topic nr
        # column 1 = weight - > the user introduces it
        # column 2 = topic description
        #####################
        self.table_widget_topic_list.clearContents()
        self.table_widget_topic_list.setRowCount(n_topics)
        self.table_widget_topic_list.setColumnCount(3)

        for i in range(n_topics):
            item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            self.table_widget_topic_list.setItem(i, 0, item_topic_nr)
            item_topic_description = QtWidgets.QTableWidgetItem(str(topic_words[i]))
            self.table_widget_topic_list.setItem(i, 2, item_topic_description)

        self.table_widget_topic_list.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table_widget_topic_list.resizeColumnsToContents()
        return

    def clicked_get_topic_list(self):
        # @TODO
        return

