# -*- coding: utf-8 -*-
"""
A generic class to manage the user navigation through a multilevel options
menu.

Created on March. 04, 2019

@author: Jesús Cid Sueiro
         Based on former menu manager scripts by Jerónimo Arenas.
"""

import os
import copy
import platform
import yaml


class MenuNavigator(object):
    """
    A class to manage the user navigation through a multilevel options menu.

    The structure of multilevel menu options with their associated
    actions and parameters should be defined in a yaml file.

    A taskmanager class is required to take the selected actions with the
    given parameters.

    """

    def __init__(self, tm, path2menu, paths2data=None):
        """
        Initializes a menu navigator.

        Parameters
        ----------
        tm : object
            A task manager object, that will be in charge of executing all
            actions selected by the user through the menu interaction. Thus, it
            must contain:
            (1) One action method per method specified in the menu structure
            (2) Data collection methods, required for some menus with dynamic
            options.

        path2menu : str
            The route to the yaml file containing the menu structure

        paths2data : dict or None, optional (default=None)
            A dictionary of paths to data repositories. The key is a name of
            the path, and the value is the path.
        """

        self.tm = tm                   # Task manager object
        self.paths2data = paths2data   # Dictionary of paths to input data
        self.path2menu = path2menu     # Location of the menu description file

        return

    def clear(self):
        """
        Cleans terminal window
        """

        # Checks if the application is running on windows or other OS
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

        return

    def query_options(self, options, active_options=None, msg=None,
                      zero_option='exit'):
        """
        Prints a heading and the subset of options indicated in the list of
        active_options, ask the user and returns the user selection

        Parameters
        ----------
        options : dict
            A dictionary of options
        active_options : list or None, optional (default=None)
            List of option keys indicating the available options to print.
            If None, all options are shown.
        msg : str or None, optional (default=None)
            Heading message to be printed before the list of available options
        zero_option : str {'exit', 'up'}, optional (default='exit')
            If 'exit', an exit option is shown
            If 'up', an option to go back to the main menu

        Returns
        -------
        option : str
            Selected option
        """

        # Print the heading messsage
        if msg is None:
            print('\n')
            print('**************')
            print('*** MAIN MENU.')
            print('Available options:')
        else:
            print(msg)

        # Print the active options
        if active_options is None:
            # If no active options ar specified, all of them are printed.
            active_options = list(options.keys())

        for n, opt in enumerate(active_options):
            print(' {0}. {1}'.format(n + 1, options[opt]))

        n_opt = len(active_options)
        if zero_option == 'exit':
            print(' 0. Exit the application\n')
            n_opt += 1
        elif zero_option == 'up':
            print(' 0. Back to menu\n')
            n_opt += 1

        range_opt = range(n_opt)

        n_option = None
        while n_option not in range_opt:
            n_option = input(f'What would you like to do? '
                             f'[{range_opt[0]}-{range_opt[-1]}]: ')
            try:
                n_option = int(n_option)
            except:
                print('Write a number')
                n_option = None

        if n_option == 0:
            option = 'zero'
        else:
            option = active_options[n_option - 1]
        return option

    # This method is not used in the current version of this class. Consider
    # removing it.
    def request_confirmation(self, msg="     Are you sure?"):
        """
        Request confirmation from user

        Parameters
        ----------
        msg : str, optional (default="    Are you sure?")
            Message printed to request confirmation

        Returns
        -------
        r : str {'yes', 'no'}
            User respones
        """

        # Iterate until an admissible response is got
        r = ''
        while r not in ['yes', 'no']:
            r = input(msg + ' (yes | no): ')

        return r == 'yes'

    def front_page(self, title):
        """
        Prints a simple title heading the application user screen

        Parameters
        ----------
        title : str
            Title message to be printed
        """

        self.clear()

        n = len(title)
        print('*' * (n + 8))
        print(f'*** {title} ***')
        print('*' * (n + 8))
        print("")

        return

<<<<<<< HEAD
    def navigate(self, option=None, active_options=None):

=======
    def get_options(self, tasks_only=False):
        """
        Returns a list of all possible actions.

        Parameters
        ----------
        tasks_only : boolean, optional (default=True)
            If true, only options that correspond to task names (methods from
            the task_manager class) are returned

        Returns
        -------
        options : list of tuple (str, str)
            A list of possible action.
            Is action is a tuple (name_of_the_action, title), where title is
            the description of the action taken from the options_menu file.
        """

        with open(self.path2menu, 'r', encoding='utf8') as f:
            menu = yaml.safe_load(f)

        if tasks_only:
            no_tasks = [k for k, v in menu.items() if k != 'root'
                        and 'options' in v
                        and isinstance(v['options'], list)
                        and len(v['options']) > 0
                        and isinstance(v['options'][0], str)]
            options = [
                (x, menu[x]['title']) for x in menu if x != 'root'
                and x not in no_tasks]

        else:
            options = [(x, menu[x]['title']) for x in menu if x != 'root']

        return options

    def navigate(self, option=None, active_options=None, iterate=True):
>>>>>>> origin/main
        """
        Manages the menu navigation loop

        Parameters
        ----------
        option : str, optional (default=None)
            Initial selection from the menu of options.
            If None, the initial selection will be requested to the user
        active_options : list or None, optional (default=None)
            List of option keys indicating the available options to print.
            If None, all options are shown.
        iterate : boolean, optional (default=True)
            If true, a sequence of menu options is shown to the
            user until the zero (exit) menu option is selected
            by the user
            If false, the
        """

        # #####################
        # Main interaction loop

        var_exit = False

        # ########################
        # Prepare user interaction
        # ########################

        # This is the complete list of level-0 options.
        # The options that are shown to the user will depend on the project
        # state
        with open(self.path2menu, 'r', encoding='utf8') as f:
            menu = yaml.safe_load(f)

        default_opt = menu['root']['options']
        options_all = [x for x in menu if x != 'root']
        opt_dict = {x: menu[x]['title'] for x in options_all}

        query_needed = option is None
        zero_opt = 'exit'

        # ################
        # Interaction loop
        # ################
        while not var_exit:

            # Query an option to the user if needed
            if query_needed:
                option = self.query_options(
                    opt_dict, active_options, zero_option=zero_opt)
            else:
                # From now on, the query will be always needed inside the loop.
                query_needed = True

            if option == 'zero':
                # Activate flag to exit the application
                if zero_opt == 'exit':
                    var_exit = True
                else:
                    active_options = copy.copy(default_opt)
                    zero_opt = 'exit'

            elif ('options' in menu[option]
                  and not isinstance(menu[option]['options'][0], dict)):

                # type(menu[option]['options'][0]) != dict):
                # Select new options to query
                active_options = menu[option]['options']
                zero_opt = 'up'

            else:

                # Default dictionary of arguments is the empty dictionary
                if 'options' in menu[option]:
                    opts = menu[option]['options']
                else:
                    opts = {}

                # The option contains at least one or more arguments that
                # should be selected by the user.

                # Initialize list of parameters (arguments) for the selected
                # method
                all_params = []
                param = None

                # Select parameers for the selected method
                # for type_arg, arg in menu[option]['options'].items():
                for opt in opts:

                    type_arg, arg = list(*opt.items())

                    if param == 'zero':
                        break

                    if type_arg[:10] == 'parameters':

                        # Query parameter to user from the values given in the
                        # menu
                        if isinstance(arg, list):   # type(arg) == list:
                            param_opts = {p: p for p in arg}
                        else:
                            param_opts = copy.copy(arg)
                        param = self.query_options(param_opts,
                                                   zero_option='up')
                        all_params.append(param)

                    elif type_arg[:10] == 'get_method':

                        # Get parameters from the method specified in the
                        # get_method field of the menu (now in variable arg)
                        # and with the current parameters (in all_params)
                        parameters = getattr(self.tm, arg)(*all_params)
                        param_opts = {p: p for p in parameters}
                        param = self.query_options(
                            param_opts, zero_option='up')
                        all_params.append(param)

                    elif type_arg[:4] == 'path':

                        # The options are in the following path:
                        path2opts = self.paths2data[arg]
                        # Read and query parameter options
                        files_and_folders = [x for x in os.listdir(path2opts)
                                             if x != '.DS_Store']

                        param_opts = {f: f for f in files_and_folders}
                        param = self.query_options(param_opts,
                                                   zero_option='up')
                        if param != 'zero':
                            param = os.path.join(path2opts, param)
                            all_params.append(param)

                    else:

                        exit(f"ERROR: Unknown type of option '{type_arg}'")

                # Call the method specified in option1
                if param != 'zero':
                    if 1 == 2:
                        # Call method with the selected parameter
                        getattr(self.tm, option)(*all_params)

                        # Update list of active menu options
                        if 'post_opts' in menu[option]:
                            active_options = menu[option]['post_opts']
                        else:
                            active_options = copy.copy(default_opt)
                    else:
                        # Call method with the selected parameter
                        while True:
                            parameters = getattr(self.tm, option)(*all_params)
                            try:
                                if len(parameters) == 0:
                                    break
                            except:
                                break
                            param_opts = {p: p for p in parameters}
                            param = self.query_options(
                                param_opts, zero_option='up')
                            if param == 'zero':
                                break
                            all_params = [param]

                        # Update list of active menu options
                        if 'post_opts' in menu[option]:
                            active_options = menu[option]['post_opts']
                        else:
                            active_options = copy.copy(default_opt)

                else:
                    # Activate flag to exit the application
                    active_options = copy.copy(default_opt)

                zero_opt = 'exit'

                var_exit = var_exit or not iterate

        return
