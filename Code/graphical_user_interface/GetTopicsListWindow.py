# -*- coding: utf-8 -*-
"""
Class representing the window for controlling the getting of a subcorpus from a given list of topics

@author: lcalv
"""

from PyQt5 import uic, QtWidgets
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
        self.selectedTag = None
        self.tw = None
        self.T = None
        self.df_metadata = None
        self.n_max = 0
        self.s_min = 0

        # INFORMATION BUTTONS
        ########################################################################
        self.info_button_topic_list.setIcon(QIcon('Images/help2.png'))
        self.info_button_topic_list.setToolTip(Messages.INFO_TOPIC_LIST)
        self.info_button_selected_tag.setIcon(QIcon('Images/help2.png'))
        self.info_button_selected_tag.setToolTip(Messages.INFO_TAG)

        self.get_topic_list_push_button.clicked.connect(self.clicked_get_topic_list)

    def show_topics(self):
        topic_words, self.T, self.df_metadata, self.n_max, self.s_min = self.tm.get_topic_words_gui()
        n_topics = len(topic_words)
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
            item_topic_description = QtWidgets.QTableWidgetItem(str(topic_words[i]))
            self.table_widget_topic_list.setItem(i, 1, item_topic_description)

        # self.table_widget_topic_list.setSizeAdjustPolicy(
        #    QtWidgets.QAbstractScrollArea.AdjustToContents)
        # self.table_widget_topic_list.resizeColumnsToContents()
        return

    def clicked_get_topic_list(self):
        if self.line_topic_list.text() is None:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.NO_TOPIC_LIST_SELECTED)
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
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.NO_TAG_SELECTED)
        else:
            self.selectedTag = str(self.line_edit_get_tag.text())

        self.hide()
        self.line_topic_list.setText("")
        self.line_edit_get_tag.setText("")
        return

