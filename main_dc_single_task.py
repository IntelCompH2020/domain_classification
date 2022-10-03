#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program
Created on Sep 06 2022
@author: Jesús Cid-Sueiro
"""

import pathlib
import argparse

# Local imports
from src.menu_navigator.menu_navigator import MenuNavigator
from src.task_manager import TaskManagerCMD


# ########################
# Main body of application
def main():

    # ################
    # Get menu options

    # Get the menu options from a options_menu file.
    path2menu = pathlib.Path('config', 'options_menu.yaml')
    # This is a fake menu because it is used only to get the list of options
    options = MenuNavigator(None, path2menu).get_options()
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
    args = parser.parse_args()

    # Create task manager object
    project_path = pathlib.Path(args.p)
    tm = TaskManagerCMD(project_path, path2source=args.source,
                        path2zeroshot=args.zeroshot)

    # #####################
    # Run preparation tasks

    # Load or create project
    if project_path.is_dir():
        tm.load()
    else:
        tm.create()
    print("\n*** END.\n")

    # Initialize menu navigator
    # FIXME: The paths2data dict is likely unused, maybe it could be ignored
    paths2data = {'input': pathlib.Path('example_folder', 'input'),
                  'imported': pathlib.Path('example_folder', 'imported')}
    menu = MenuNavigator(tm, path2menu, paths2data)
    active_options = None

    # Load labels if the task requires it
    options_needing_labels = {
        'PU_learning', 'get_feedback', 'update_model',
        'import_export_annotations', 'evaluate_PUlabels', 'train_PUmodel',
        'evaluate_PUmodel', 'performance_metrics_PU', 'retrain_model',
        'reevaluate_model', 'performance_metrics_PN', 'import_annotations',
        'export_annotations'}

    option = args.task
    if option in options_needing_labels:
        menu.navigate('load_labels', active_options, iterate=False)

    # ########
    # Run task

    menu.navigate(option, active_options, iterate=False)


# ############
# Execute main
if __name__ == '__main__':
    main()
