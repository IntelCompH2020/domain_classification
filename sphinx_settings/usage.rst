Getting Started with Domain Classifier
======================================

.. meta::
   :description lang=en: How to use the domain classifier.

There are two different versions of the Domain Classifier. They differ in the way to manage the user interaction:

  1. ``main_gui.py``: user interacts through a Graphical User Interface (GUI).
  2. ``main_domain_classifier.py``: user interacts through a terminal (command window).


Graphical User Interface
------------------------

You will need a terminal window to launch the GUI: open a terminal, go to the roof folder of the software and run

.. code-block:: bash

    $ python main_gui.py --p /path/to/project  
                         --source /path/to/datasets  
                         —-zeroshot /path/to/zeroshot

where

   * ``/path/to/project`` is the path to a new or an existing project in which the application’s output will be saved.
   * ``/path/to/datasets`` is the path to the source data folder.
   * ``/path/to/zeroshot`` is the path to a folder containing a pre-trained zero-shot model utilized for the selection of a subcorpus from a category name.

All these parameters are optional. For instance, you can simply run

.. code-block:: bash

   $ python main_gui.py

and select the appropriate folders inside the GUI.


Terminal mode
-------------

To use the interaction through a command window, open a terminal, go to the root folder of the software and run

.. code-block:: bash

    $ python main_domain_classifier.py --p /path/to/project  
                                       --source /path/to/datasets
                                       —-zeroshot /path/to/zeroshot

where

   * ``/path/to/project`` is the path to a new or an existing project in which the application’s output will be saved.
   * ``/path/to/datasets`` is the path to the source data folder.
   * ``/path/to/zeroshot`` is the path to a folder containing a pre-trained zero-shot model utilized for the selection of a subcorpus from a category name.



