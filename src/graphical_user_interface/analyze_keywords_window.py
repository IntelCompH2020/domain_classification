# -*- coding: utf-8 -*-
"""
Class representing the window for controlling the analysis of the presence of
selected keywords in the corpus

@author: lcalv
"""
import numpy as np
from PyQt5 import uic, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from src.graphical_user_interface.messages import Messages


# @ TODO: Add wt as parameter configuration in GUI?

class AnalyzeKeywordsWindow(QtWidgets.QDialog):
    def __init__(self, tm):
        super(AnalyzeKeywordsWindow, self).__init__()

        # Load UI and configure default geometry of the window
        # ####################################################################
        uic.loadUi("UIS/analyze_keywords.ui", self)
        self.initUI()

        # ATTRIBUTES
        # ####################################################################
        self.tm = tm
        self.figure = plt.figure()
        # this is the Canvas Widget that displays the 'figure'
        # It takes the 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.layoutPlot.addWidget(self.canvas)

    def initUI(self):
        self.setWindowIcon(QIcon('Images/dc_logo.png'))
        self.setWindowTitle(Messages.WINDOW_TITLE)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def do_analysis(self):
        y, df_stats, kf_stats = self.tm.analyze_keywords()
        n_top = 25

        # Plot sorted document scores
        self.figure.clear()
        ax = self.figure.add_subplot(311)
        ax.plot(sorted(y))
        ax.set_title("Sorted document scores")
        ax.set_xlabel('Document')
        ax.set_ylabel('Score')

        sorted_stats = sorted(df_stats.items(), key=lambda item: -item[1])
        hot_tokens, hot_values = zip(*sorted_stats[n_top::-1])
        y_pos = np.arange(len(hot_tokens))
        ax2 = self.figure.add_subplot(312)
        ax2.barh(hot_tokens, hot_values, align='center', alpha=0.4)
        ax2.set_yticks(y_pos, hot_tokens)
        ax2.set_title("Document frequencies")
        ax2.set_xlabel('No. of docs')

        sorted_stats = sorted(kf_stats.items(), key=lambda item: -item[1])
        hot_tokens, hot_values = zip(*sorted_stats[n_top::-1])
        y_pos = np.arange(len(hot_tokens))
        ax3 = self.figure.add_subplot(313)
        ax3.barh(hot_tokens, hot_values, align='center', alpha=0.4)
        ax3.set_yticks(y_pos, hot_tokens)
        ax3.set_title("Keyword frequencies")
        ax3.set_xlabel('No. of keywords')

        self.canvas.draw()

        return
