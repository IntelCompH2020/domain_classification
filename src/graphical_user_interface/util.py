# -*- coding: utf-8 -*-
"""
@author: lcalv
"""

# General imports
from PyQt5 import QtCore

# Local imports
from src.graphical_user_interface.worker import Worker


def toggle_menu(gui, max_width):
    """
    Method to control the movement of the Toggle menu located on the
    left. When collapsed, only the icon for each of the options is shown;
    when expanded, both icons and name indicating the description of the
    functionality are shown.
    Based on the code available at:
    https://github.com/Wanderson-Magalhaes/Toggle_Burguer_Menu_Python_PySide2/blob/master/ui_functions.py

    Parameters
    ----------
    gui: MainWindow
        MainWindow object to which the toggle menu will be appended.
    maxWidth: int
        Maximum width to which the toggle menu is going to be expanded.
    """

    # Get width
    width = gui.frame_left_menu.width()
    max_extend = max_width
    standard = 70

    # Set maximum width
    if width == standard:
        width_extended = max_extend
        # Show texts instead of the icons when the maximum width of the toggle bar is set
        gui.pushButtonLoad.setText('Corpus / labels')
        gui.pushButtonTrain.setText('Train PU model')
        gui.pushButtonGetFeedback.setText('Get feedback')
        gui.label_logo.setFixedSize(width_extended, width_extended)

    else:
        # Remove texts and place the icons when the toggle bar is set to its default width
        width_extended = standard
        gui.pushButtonLoad.setText('')
        gui.pushButtonTrain.setText('')
        gui.pushButtonGetFeedback.setText('')
        gui.label_logo.setFixedSize(width_extended, width_extended)

    # Configure movement of the toggle bar
    gui.animation = QtCore.QPropertyAnimation(
        gui.frame_left_menu, b"minimumWidth")
    gui.animation.setDuration(400)
    gui.animation.setStartValue(width)
    gui.animation.setEndValue(width_extended)
    gui.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
    gui.animation.start()


def execute_in_thread(gui, function, function_output, progress_bar):
    """
    Method to execute a function in the secondary thread while showing
    a progress bar at the time the function is being executed if a progress bar object is provided.
    When finished, it forces the execution of the method to be
    executed after the function executing in a thread is completed.
    Based on the functions provided in the manual available at:
    https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/

    Parameters
    ----------
    function: UDF
        Function to be executed in thread
    function_output: UDF
        Function to be executed at the end of the thread
    progress_bar: QProgressBar
        If a QProgressBar object is provided, it shows a progress bar in the
        main thread while the main task is being carried out in a secondary thread
    """

    # Pass the function to execute
    gui.worker = Worker(function)

    # Show progress if a QProgressBar object has been passed as argument to the function
    if progress_bar is not None:
        signal_accept(progress_bar)

    # Connect function that is going to be executed when the task being
    # carrying out in the secondary thread has been completed
    gui.worker.signals.finished.connect(function_output)

    # Execute
    gui.thread_pool.start(gui.worker)


def signal_accept(progress_bar):
    """
    Makes the progress bar passed as an argument visible and configures it for
    an event whose duration is unknown by setting both its minimum and maximum
    both to 0, thus the bar shows a busy indicator instead of a percentage of steps.

    Parameters
    ----------
    progress_bar: QProgressBar
         Progress bar object in which the progress is going to be displayed.
    """
    progress_bar.setVisible(True)
    progress_bar.setMaximum(0)
    progress_bar.setMinimum(0)


def change_background_color_text_edit(text_edit, prediction):
    """
    Method to that changes the border color of the QTextEdit associated with the documents to be annotated, based on
    the predicted class selected by the user.

    Parameters
    ----------
    text_edit: QTextEdit
        QTextEdit (document) whose border color is going to be updated based on the
    prediction: int (0 or 1)
        Predicted class specified by the user. If prediction == 1, the document's associated QTextEdit's border
        color is set to #6A7288; if prediction == 0, it is set to #DB8678.
    """
    if prediction == 1:
        text_edit.setStyleSheet("""
            QTextEdit {	
                background-color: #FFFFFF;
                border-radius: 5px;
                gridline-color: #FFFFFF;
                border-bottom: 1px solid #FFFFFF;
                border: 6px solid #6A7288;
            }
            QScrollBar:vertical {
                border: none;
                background: #6A7288;
                width: 2px;
                margin: 21px 0 21px 0;
                border-radius: 0px;
            }
        """)
    elif prediction == 0:
        text_edit.setStyleSheet("""
            QTextEdit {	
                background-color: #FFFFFF;
                border-radius: 5px;
                gridline-color: #FFFFFF;
                border-bottom: 1px solid #FFFFFF;
                border: 6px solid #DB8678;
            }
            QScrollBar:vertical {
                border: none;
                background: #DB8678;
                width: 2px;
                margin: 21px 0 21px 0;
                border-radius: 0px;
            }
        """)
