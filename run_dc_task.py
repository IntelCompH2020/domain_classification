#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program
Created on Sep 06 2022
@author: Jesús Cid-Sueiro
"""

import pathlib
import argparse
import inspect
import yaml

# Local imports
from src.menu_navigator.menu_navigator import MenuNavigator
from src.task_manager import TaskManager      # , TaskManagerCMD


# ########################
# Main body of application
def main():

    # ################
    # Get menu options

    # Get the menu options from a options_menu file.
    path2menu = pathlib.Path('config', 'options_menu.yaml')
    # This is a fake menu because it is used only to get the list of options
    options = MenuNavigator(None, path2menu).get_options(tasks_only=True)
    # option_names = [x[0] for x in options]
    options_txt = "\n".join([f'  - {x[0]}:  {x[1]}' for x in options])

    # ####################
    # Read input arguments

    # Read input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--task', required=True,
        help=f"Command/task to be executed. It must be one of the following:"
             f"\n{options_txt}")
    parser.add_argument(
        '--p', required=True,
        help="path to a new or an existing project")
    parser.add_argument(
        '--source', default="../datasets",
        help="path to the source data folder")
    parser.add_argument(
        '--zeroshot', default='../zero_shot_model/Sciro-Shot',
        help="path to the zero-shot model folder")
    parser.add_argument(
        '--class_name',
        help="Name of the labeled dataset")
    args, other_args = parser.parse_known_args()

    # ################################################
    # Read input arguments for the task manager method

    # This if is used to avoid the cases which would re-read parameter
    # class_name, because it has been already read, raising an error
    if args.task not in {'load_labels', 'reset_labels'}:
        params = inspect.getfullargspec(getattr(TaskManager, args.task))
        n_params = len(params.args)
        n_defaults = 0 if params.defaults is None else len(params.defaults)

        # Index starts from 1 because argument 0 is 'self'
        arg_names = params.args[1:n_params - n_defaults]
        kwarg_names = params.args[n_params - n_defaults:]
        default_values = [] if n_defaults == 0 else params.defaults

        for arg in arg_names:
            arg_type = str    # Default
            if arg in params.annotations:
                arg_type = params.annotations[arg]
            parser.add_argument(f'--{arg}', type=arg_type, required=True)

        for arg, value in zip(kwarg_names, default_values):
            arg_type = str    # Default
            if arg in params.annotations:
                arg_type = params.annotations[arg]
            parser.add_argument(f'--{arg}', type=arg_type, default=value)

        args = parser.parse_args()

    # Create task manager object
    with open('config/parameters.default.yaml', 'r', encoding='utf8') as f:
        parameter_default = yaml.safe_load(f)
    project_path = pathlib.Path(parameter_default['project_folder_path']) / pathlib.Path(args.p)


    # tm = TaskManagerCMD(project_path, path2source=args.source,
    #                     path2zeroshot=args.zeroshot)
    tm = TaskManager(project_path, path2source=args.source,
                     path2zeroshot=args.zeroshot)

    # #####################
    # Run preparation tasks

    # If the task is load or create, we simply need to run it.
    if args.task not in {'create', 'load'}:

        # Load or create project
        if project_path.is_dir():
            tm.load()
        else:
            tm.create()

        # Load labels if the task requires it
        options_needing_labels = {
            'load_labels', 'PU_learning', 'get_feedback', 'update_model',
            'import_export_annotations', 'evaluate_PUlabels', 'train_PUmodel',
            'evaluate_PUmodel', 'performance_metrics_PU', 'sample_documents',
            'get_labels_from_docs', 'annotate', 'retrain_model',
            'reevaluate_model', 'performance_metrics_PN', 'import_annotations',
            'export_annotations'}

        option = args.task
        if option in options_needing_labels:
            if args.class_name is not None:
                tm.load_labels(args.class_name)
            else:
                raise TypeError(
                    f"Task {args.task} requires argument --class_name")

    # ########
    # Run task
    if args.task not in {'load_labels', 'reset_labels'}:

        # Get args
        arg_values = [getattr(args, name) for name in arg_names]

        # Get kwargs
        # - default values
        kwargs = dict(zip(kwarg_names, default_values))
        # - user-defined values
        for arg in kwarg_names:
            if arg in args:
                kwargs[arg] = getattr(args, arg)

        # Run task
        getattr(tm, args.task)(*arg_values, **kwargs)

    # If the task is reset_labels run it with the class name.
    elif args.task == 'reset_labels':
        tm.reset_labels(args.class_name)

    # The case load_labels is ignored, because that task has been already done.

    # TEST PENDING:
    # - get_feedback:  Get relevance feedback from user
    # - evaluate_PUlabels:  Evaluate subcorpus with respect to a gold standard
    # - evaluate_PUmodel:  Evaluate PU classifier model with the available labs
    # - retrain_model:  Retrain model with manual annotations
    # - reevaluate_model:  Evaluate retrained model.
    # - performance_metrics_PN:  Show all performance metrics
    # - import_annotations:  Import annotations (overwrites existing annots)
    # - export_annotations:  Export annotations (delete older annot files)

    print("\n*** END.\n")

    return


# ############
# Execute main
if __name__ == '__main__':
    main()
