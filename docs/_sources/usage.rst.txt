Getting Started with Domain Classifier
======================================

.. meta::
   :description lang=en: How to use the domain classifier.

There are different scritps available to run the Domain Classifier. They differ in the way to manage the user interaction:

  1. ``main_gui.py``: user interacts through a Graphical User Interface (GUI).
  2. ``main_domain_classifier.py``: user interacts through a terminal (command window).
  3. ``main_dc_single_task.py``: to execute a single task (requires user interaction).
  4. ``run_dc_task.py``: to execute a single task (without user interaction).


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


Single task execution with user interaction
-------------------------------------------

To use the interaction through a command window, open a terminal, go to the root folder of the software and run

.. code-block:: bash

    $ python main_dc_single_task.py --p /path/to/project  
                                    --source /path/to/datasets
                                    —-zeroshot /path/to/zeroshot
                                    --task TASK

where

   * ``/path/to/project`` is the path to a new or an existing project in which the application’s output will be saved.
   * ``/path/to/datasets`` is the path to the source data folder.
   * ``/path/to/zeroshot`` is the path to a folder containing a pre-trained zero-shot model utilized for the selection of a subcorpus from a category name.
   * TASK is the name of the task to run.

You can run

.. code-block:: bash

    $ python main_dc_single_task.py --h

to see available tasks.


Single task execution without user interaction
----------------------------------------------

To run a specific task, all parameter of the task should be introduced through the comman window. To do so, you can run

.. code-block:: bash

    $ python run_dc_task.py --p /path/to/project  
                            --source /path/to/datasets
                            —-zeroshot /path/to/zeroshot
                            --task TASK
                            --param1 PARAM1
                            --param1 PARAM2
                            ...

where

   * ``/path/to/project`` is the path to a new or an existing project in which the application’s output will be saved.
   * ``/path/to/datasets`` is the path to the source data folder.
   * ``/path/to/zeroshot`` is the path to a folder containing a pre-trained zero-shot model utilized for the selection of a subcorpus from a category name.
   * TASK is the name of the task to run.
   * param1, param2 are the names of the specific parameters
   * PARAM1, PARAM2 are their values

You can run

.. code-block:: bash

    $ python main_dc_single_task.py --h

to see available tasks.

