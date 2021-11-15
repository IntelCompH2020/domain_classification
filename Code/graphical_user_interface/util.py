# -*- coding: utf-8 -*-
"""
Class with a set of auxiliary functions for the GUI deployment.

@author: lcalv
"""
import os
from PyQt5 import QtCore, QtWidgets, QtGui


def toggleMenu(gui, maxWidth):
    """Method to control the movement of the Toggle menu located on the
    left. When collapsed, only the icon for each of the options is shown;
    when expanded, both icons and name indicating the description of the
    functionality are shown.

    Parameters:
    ----------
    * gui       -  GUI object to which the toggle menu will be appended.
    * maxWidth  -  Maximum width to which the toggle menu is going to be
                   expanded.
    """
    # GET WIDTH
    width = gui.frame_left_menu.width()
    maxExtend = maxWidth
    standard = 70

    # SET MAX WIDTH
    if width == 70:
        widthExtended = maxExtend
        # SHOW TEXT INSTEAD OF ICON
        gui.pushButtonLoad.setText('Configuration')
        gui.label_logo.setFixedSize(widthExtended, widthExtended)

    else:
        widthExtended = standard
        gui.pushButtonLoad.setText('')
        gui.label_logo.setFixedSize(widthExtended, widthExtended)

    # ANIMATION
    gui.animation = QtCore.QPropertyAnimation(gui.frame_left_menu, b"minimumWidth")
    gui.animation.setDuration(400)
    gui.animation.setStartValue(width)
    gui.animation.setEndValue(widthExtended)
    gui.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
    gui.animation.start()

