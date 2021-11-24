# -*- coding: utf-8 -*-
"""
Class representing the window for controlling the getting of a subcorpus from a given list of keywords

@author: lcalv
"""

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget

from Code.graphical_user_interface.Messages import Messages


class GetKeywordsWindow(QtWidgets.QDialog):
    def __init__(self, tm):
        super(GetKeywordsWindow, self).__init__()

        # Load UI and configure default geometry of the window
        ########################################################################
        uic.loadUi("UIS/get_labels_by_keywords.ui", self)
        self.initUI()

        # ATTRIBUTES
        #######################################################################
        self.tm = tm
        self.selectedKeywords = None
        self.selectedTag = None

        # INFORMATION BUTTONS
        ########################################################################
        self.info_button_selected_keywords.setIcon(QIcon('Images/help2.png'))
        self.info_button_selected_keywords.setIconSize(0.75 * QSize(self.info_button_selected_keywords.width(),
                                                                    self.info_button_selected_keywords.height()))
        self.info_button_selected_keywords.setToolTip(Messages.INFO_TYPE_KEYWORDS)
        self.info_button_selected_tag.setIcon(QIcon('Images/help2.png'))
        self.info_button_selected_tag.setIconSize(0.75 * QSize(self.info_button_selected_tag.width(),
                                                               self.info_button_selected_tag.height()))
        self.info_button_selected_tag.setToolTip(Messages.INFO_TAG)

        self.get_labels_push_button.clicked.connect(self.clicked_select_keywords)

    def initUI(self):
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def show_suggested_keywords(self):
        suggested_keywords = self.tm.get_suggested_keywords_gui()
        self.text_edit_show_keywords.setPlainText(suggested_keywords)

    def clicked_select_keywords(self):
        if self.line_edit_get_keywords.text() is None:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.NO_KEYWORDS_SELECTED)
        else:
            keywords = self.line_edit_get_keywords.text()
            # Split by commas, removing leading and trailing spaces
            _keywords = [x.strip() for x in keywords.split(',')]
            # Remove multiple spaces
            _keywords = [' '.join(x.split()) for x in _keywords]
            self.selectedKeywords = _keywords

        if self.line_edit_get_tag.text() is None:
            QtWidgets.QMessageBox.warning(self, Messages.DC_MESSAGE, Messages.NO_TAG_SELECTED)
        else:
            self.selectedTag = str(self.line_edit_get_tag.text())
        self.hide()
        self.text_edit_show_keywords.setPlainText("")
        self.line_edit_get_keywords.setText("")
        self.line_edit_get_tag.setText("")
        return

