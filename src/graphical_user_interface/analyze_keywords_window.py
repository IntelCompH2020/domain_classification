# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                   CLASS ANALYZE KEYWORDS WINDOW                        ***
******************************************************************************
Class representing the window that is used for the analysis of the presence of
selected keywords in the corpus
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
# General imports
import numpy as np
from PyQt5 import uic, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Local imports
from src.graphical_user_interface.messages import Messages


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
        # This is the Canvas Widget that displays the 'figure'
        # It takes the 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        self.layoutPlot.addWidget(self.canvas)

    def initUI(self):
        """Configures the elements of the GUI window that are not configured in the UI, i.e. icon of the application, the application's title, and the position of the window at its opening.
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

    def do_analysis(self):
        """Performs the analysis of the keywords based by showing the "Sorted document scores", "Document frequencies" and "Keyword frequencies" graphs.
        """        
        y, df_stats, kf_stats = self.tm.analyze_keywords()
        n_top = 25

        # Plot sorted document scores
        self.figure.clear()
        ax = self.figure.add_subplot(311)
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.tick_params(axis='both', which='minor', labelsize=4)
        ax.plot(sorted(y))
        ax.set_title("Sorted document scores", fontsize=6)
        ax.set_xlabel('Document',  fontsize=4)
        ax.set_ylabel('Score',  fontsize=4)

        sorted_stats = sorted(df_stats.items(), key=lambda item: -item[1])
        hot_tokens, hot_values = zip(*sorted_stats[n_top::-1])
        y_pos = np.arange(len(hot_tokens))
        ax2 = self.figure.add_subplot(312)
        ax2.tick_params(axis='both', which='major', labelsize=4)
        ax2.tick_params(axis='both', which='minor', labelsize=4)
        ax2.barh(hot_tokens, hot_values, align='center', alpha=0.4)
        ax2.set_yticks(y_pos, hot_tokens)
        ax2.set_title("Document frequencies", fontsize=6)
        ax2.set_xlabel('No. of docs', fontsize=4)

        sorted_stats = sorted(kf_stats.items(), key=lambda item: -item[1])
        hot_tokens, hot_values = zip(*sorted_stats[n_top::-1])
        y_pos = np.arange(len(hot_tokens))
        ax3 = self.figure.add_subplot(313)
        ax3.tick_params(axis='both', which='major', labelsize=4)
        ax3.tick_params(axis='both', which='minor', labelsize=4)
        ax3.barh(hot_tokens, hot_values, align='center', alpha=0.4)
        ax3.set_yticks(y_pos, hot_tokens)
        ax3.set_title("Keyword frequencies", fontsize=6)
        ax3.set_xlabel('No. of keywords', fontsize=4)

        self.canvas.draw()

        return
