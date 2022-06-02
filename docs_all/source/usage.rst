Getting Started with Domain Classifier
======================================

.. meta::
   :description lang=en: How to use the domain classifier.

There are two different versions of the Domain Classifier. They differ in the way to manage the user interaction:

  1. main_domain_classifier.py: user interacts through a terminal (command window).
  2. main_gui.py: user interacts through a Graphical User Interface (GUI).


Terminal mode
-------------

To use te interaction through a command window, open a terminal, go to the root folder of the software and run

.. prompt:: bash $

    python main_domain_classifier.py --p [project_folder]  --source [dataset_folder]  —-zeroshot [zero_shot_folder]

where

   * [project_folder] is the path to a new or an existing project in which the application’s output will be saved.
   * [dataset_folder] is the path to the source data folder.
   * [zero_shot_folder] is the path to a folder containing a pre-trained zero-shot model utilized for the selection of a subcorpus from a category name.


Graphical User Interface
------------------------

You will need a terminal window to launch the GUI: open a terminal, go to the roof folder of the software and run

.. prompt:: bash $

    python main_gui.py --p [project_folder]  --source [dataset_folder]  —-zeroshot [zero_shot_folder]

where

   * [project_folder] is the path to a new or an existing project in which the application’s output will be saved.
   * [dataset_folder] is the path to the source data folder.
   * [zero_shot_folder] is the path to a folder containing a pre-trained zero-shot model utilized for the selection of a subcorpus from a category name.

All these parameters are optional. For instance, you can simply run

.. prompt:: bash $

   python main_gui.py

and select the appropriate folders inside the GUI.


Other info about  Sphinx
=========================


* Generate web pages, printable PDFs, documents for e-readers (ePub),
  and more all from the same sources
* You can use reStructuredText or :ref:`Markdown <intro/getting-started-with-sphinx:Using Markdown with Sphinx>`
  to write documentation
* An extensive system of cross-referencing code and documentation
* Syntax highlighted code samples
* A vibrant ecosystem of first and third-party :doc:`extensions <sphinx:usage/extensions/index>`

If you want to learn more about how to create your first Sphinx project, read on.
If you are interested in exploring the Read the Docs platform using an already existing Sphinx project,
check out :doc:`/tutorial/index`.

Quick start
-----------

.. seealso:: If you already have a Sphinx project, check out our :doc:`/intro/import-guide` guide.

Assuming you have Python already, :doc:`install Sphinx <sphinx:usage/installation>`:

.. prompt:: bash $

    pip install sphinx

Create a directory inside your project to hold your docs:

.. prompt:: bash $

    cd /path/to/project
    mkdir docs

Run ``sphinx-quickstart`` in there:

.. prompt:: bash $

    cd docs
    sphinx-quickstart

This quick start will walk you through creating the basic configuration; in most cases, you
can just accept the defaults. When it's done, you'll have an ``index.rst``, a
``conf.py`` and some other files. Add these to revision control.

Now, edit your ``index.rst`` and add some information about your project.
Include as much detail as you like (refer to the :doc:`reStructuredText syntax <sphinx:usage/restructuredtext/basics>`
or `this template`_ if you need help). Build them to see how they look:

.. prompt:: bash $

    make html

Your ``index.rst`` has been built into ``index.html``
in your documentation output directory (typically ``_build/html/index.html``).
Open this file in your web browser to see your docs.

.. figure:: /_static/images/first-steps/sphinx-hello-world.png
   :figwidth: 500px
   :align: center

   Your Sphinx project is built

Edit your files and rebuild until you like what you see, then commit your changes and push to your public repository.
Once you have Sphinx documentation in a public repository, you can start using Read the Docs
by :doc:`importing your docs </intro/import-guide>`.

.. warning::

   We strongly recommend to :ref:`pin the Sphinx version <guides/reproducible-builds:pinning dependencies>`
   used for your project to build the docs to avoid potential future incompatibilities.

.. _this template: https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/#id1

Using Markdown with Sphinx
--------------------------

You can use `Markdown using MyST`_ and reStructuredText in the same Sphinx project.
We support this natively on Read the Docs, and you can do it locally:

.. prompt:: bash $

    pip install myst-parser

Then in your ``conf.py``:

.. code-block:: python

   extensions = ['myst_parser']

You can now continue writing your docs in ``.md`` files and it will work with Sphinx.
Read the `Getting started with MyST in Sphinx`_ docs for additional instructions.

.. _Getting started with MyST in Sphinx: https://myst-parser.readthedocs.io/en/latest/sphinx/intro.html
.. _Markdown using MyST: https://myst-parser.readthedocs.io/en/latest/using/intro.html

External resources
------------------

Here are some external resources to help you learn more about Sphinx.

* `Sphinx documentation`_
* :doc:`RestructuredText primer <sphinx:usage/restructuredtext/basics>`
* `An introduction to Sphinx and Read the Docs for technical writers`_

.. _Sphinx documentation: https://www.sphinx-doc.org/
.. _An introduction to Sphinx and Read the Docs for technical writers: https://www.ericholscher.com/blog/2016/jul/1/sphinx-and-rtd-for-writers/