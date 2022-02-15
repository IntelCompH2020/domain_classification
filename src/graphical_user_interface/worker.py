# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:19:34 2021
@author: lcalv
******************************************************************************
***                           CLASS WORKER                                 ***
******************************************************************************
Module that inherits from QRunnable and is used to handler worker
thread setup, signals and wrap-up. It has been created based on the analogous
class provided by:
https://www.pythonguis.com/tutorials/multithreading-pyqt-applications-qthreadpool/.
"""

##############################################################################
#                                IMPORTS                                     #
##############################################################################
# General imports
from PyQt5 import QtCore
import sys
import traceback

# Local imports
from src.graphical_user_interface.worker_signals import WorkerSignals


class Worker(QtCore.QRunnable):
    """
    Worker thread. It inherits from QRunnable to handler worker thread setup, signals and
    wrap-up.

    Parameters:
    ----------
    * callback       -  The function callback to run on this worker thread.
                        Supplied args and kwargs will be passed through to the runner.
    * callback       -  Function
    * args           -  Arguments to pass to the callback function     
    * kwargs         -  Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):

        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()


    @QtCore.pyqtSlot()
    def run(self):
        """Initialises the runner function with passed args, kwargs.
        """        

        # Retrieve args/kwargs here; and fire processing using them
        try:
            self.signals.started.emit()
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            print("final signal emitted")
            self.signals.finished.emit()  # Done
