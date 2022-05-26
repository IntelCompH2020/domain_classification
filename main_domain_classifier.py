#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program

Created on March 20 2019

@author: Jesús Cid-Sueiro
"""

import pathlib
import argparse

# Local imports
from src.menu_navigator.menu_navigator import MenuNavigator
from src.task_manager import TaskManagerCMD


# ########################
# Main body of application
# ########################
def main():
    # ####################
    # Read input arguments

    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--p', type=str,
        default="../project_folder",
        help="path to a new or an existing project")
    parser.add_argument(
        '--source', type=str,
        default="../datasets",
        help="path to the source data folder")
    parser.add_argument(
        '--zeroshot', type=str,
        default='../zero_shot_model/Sciro-Shot',
        help="path to the zero-shot model folder")
    args = parser.parse_args()

    # Read project_path
    project_path = args.p
    if args.p is None:
        while project_path is None or project_path == "":
            project_path = input('-- Write the path to the project to load or '
                                 'create: ')
    project_path = pathlib.Path(project_path)

    if project_path.is_dir():
        option = 'load'
    else:
        option = 'create'
    active_options = None
    # query_needed = False

    # Create task manager object
    tm = TaskManagerCMD(project_path, path2source=args.source,
                        path2zeroshot=args.zeroshot)

    # ########################
    # Prepare user interaction
    # ########################

    paths2data = {'input': pathlib.Path('example_folder', 'input'),
                  'imported': pathlib.Path('example_folder', 'imported')}
    path2menu = pathlib.Path('config', 'options_menu.yaml')

    # ##############
    # Call navigator
    # ##############

    menu = MenuNavigator(tm, path2menu, paths2data)
    menu.front_page(title="Domain Classifier")
    menu.navigate(option, active_options)


# ############
# Execute main
if __name__ == '__main__':
    main()
