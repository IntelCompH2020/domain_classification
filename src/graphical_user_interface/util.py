# -*- coding: utf-8 -*-
"""
Class with a set of auxiliary functions for the GUI deployment.

@author: lcalv
"""

from PyQt5 import QtCore
import time
from timeloop import Timeloop
from datetime import timedelta
from src.graphical_user_interface.worker import Worker


def toggle_menu(gui, max_width):
    """Method to control the movement of the Toggle menu located on the
    left. When collapsed, only the icon for each of the options is shown;
    when expanded, both icons and name indicating the description of the
    functionality are shown.

    Parameters:
    ----------
    * gui       -  MainWindow object to which the toggle menu will be appended.
    * maxWidth  -  Maximum width to which the toggle menu is going to be
                   expanded.
    """
    # GET WIDTH
    width = gui.frame_left_menu.width()
    max_extend = max_width
    standard = 70

    # SET MAX WIDTH
    if width == standard:
        width_extended = max_extend
        # SHOW TEXT INSTEAD OF ICON
        gui.pushButtonLoad.setText('Corpus / labels')
        gui.pushButtonTrain.setText('Train PU model')
        gui.pushButtonGetFeedback.setText('Get feedback')
        gui.label_logo.setFixedSize(width_extended, width_extended)

    else:
        width_extended = standard
        gui.pushButtonLoad.setText('')
        gui.pushButtonTrain.setText('')
        gui.pushButtonGetFeedback.setText('')
        gui.label_logo.setFixedSize(width_extended, width_extended)

    # ANIMATION
    gui.animation = QtCore.QPropertyAnimation(
        gui.frame_left_menu, b"minimumWidth")
    gui.animation.setDuration(400)
    gui.animation.setStartValue(width)
    gui.animation.setEndValue(width_extended)
    gui.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
    gui.animation.start()


def execute_in_thread(gui, function, function_output, progress_bar):
    """ Method to execute a function in the secondary thread, while showing
    an animation at the time the function is being executed if animation is
    set to true. When finished, it forces the execution of the method to be
    executed after the function executing in a thread is completed.

    Parameters:
    ----------
    * function         - Function to be executed in thread
    * function_output  - Function to be executed af te the thread
    * progress_bar     - If a QProgressBar object is provided,
                         it shows a progress bar in the main thread
                         while the main task is being carried out in
                         a secondary thread
    """

    # Pass the function to execute
    gui.worker = Worker(function)
    # Any other args, kwargs are passed to the run function

    if progress_bar is not None:
        signal_accept(progress_bar)

    gui.worker.signals.finished.connect(function_output)

    # Execute
    gui.thread_pool.start(gui.worker)


def signal_accept(progress_bar):
    progress_bar.setVisible(True)
    progress_bar.setMaximum(0)
    progress_bar.setMinimum(0)