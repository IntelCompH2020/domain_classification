#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program

Created on March 20 2019

@author: Jesús Cid Sueiro
"""

import os
import pathlib
import argparse

# Local imports
from src.menu_navigator.menu_navigator import MenuNavigator
from src.task_manager import TaskManagerCMD

# ########################
# Main body of application
# ########################

# ####################
# Read input arguments

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to a new or an existing project")
parser.add_argument('--source', type=str, default='../datasets',
                    help="path to the source data folder")
args = parser.parse_args()

# Read project_path
project_path = args.p
if args.p is None:
    while project_path is None or project_path == "":
        project_path = input('-- Write the path to the project to load or '
                             'create: ')
if os.path.isdir(args.p):
    option = 'load'
else:
    option = 'create'
active_options = None
query_needed = False

# Create task manager object
tm = TaskManagerCMD(project_path, path2source=args.source)

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
menu.front_page(title="An Application example using menuNavigator")
menu.navigate(option, active_options)
