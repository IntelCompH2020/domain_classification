# -*- coding: utf-8 -*-
"""
Class representing the window for controlling the analysis of the presence of selected keywords in the corpus

@author: lcalv
"""

from PyQt5 import uic, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from Code.graphical_user_interface.Messages import Messages


class AnalyzeKeywordsWindow(QtWidgets.QDialog):
    def __init__(self, tm):
        super(AnalyzeKeywordsWindow, self).__init__()

        # Load UI and configure default geometry of the window
        ########################################################################
        uic.loadUi("UIS/analyze_keywords.ui", self)
        self.initUI()

        # ATTRIBUTES
        #######################################################################
        self.tm = tm
        self.figure = plt.figure()
        # this is the Canvas Widget that displays the 'figure'
        # It takes the 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.layoutPlot.addWidget(self.canvas)

    def initUI(self):
        self.setGeometry(100, 60, 2000, 1600)
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def do_analysis(self):
        y = self.tm.analyze_keywords_gui()

        # Plot sorted document scores
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(sorted(y))
        ax.set_title("Sorted document scores")
        ax.set_xlabel('Document')
        ax.set_ylabel('Score')
        self.canvas.draw()
        return
