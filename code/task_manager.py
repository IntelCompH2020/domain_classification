# Local imports

# You might need to update the location of the baseTaskManager class
from .base_taskmanager import baseTaskManager


class TaskManager(baseTaskManager):
    """
    This is an example of task manager that extends the functionality of
    the baseTaskManager class for a specific example application

    This class inherits from the baseTaskManager class, which provides the
    basic method to create, load and setup an application project.

    The behavior of this class might depend on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    def __init__(self, path2project, path2source=None,
                 config_fname='parameters.yaml', metadata_fname='metadata.pkl',
                 set_logs=True):
        """
        Opens a task manager object.

        Parameters
        ----------
        path2project : str
            Path to the application project
        config_fname : str, optional (default='parameters.yaml')
            Name of the configuration file
        metadata_fname : str or None, optional (default=None)
            Name of the project metadata file.
            If None, no metadata file is used.
        set_logs : bool, optional (default=True)
            If True logger objects are created according to the parameters
            specified in the configuration file
        path2source : pathlib.Path
            Path to the source data.
        """

        super().__init__(path2project, config_fname=config_fname,
                         metadata_fname=metadata_fname, set_logs=set_logs)

        # You should modify this path here to create the dictionary with the
        # default folder structure of the proyect.
        # Names will be assumed to be subfolders of the project folder in
        # self.path2project.
        # This list can be modified within an active project by adding new
        # folders. Every time a new entry is found in this list, a new folder
        # is created automatically.
        self.f_struct = {'figures': 'figures',
                         'output': 'output'}

        return

    def show_data(self, parameter):

        return

    def reset_features(self):

        return

    def get_labels(self):

        labels = ['lab1', 'lab2', 'lab3']

        return labels

    def reset_labels(self, label):

        return

    def get_images(self):

        images = ['im1', 'im2', 'im3']

        return images

    def reset_images(self, image):

        return

    def import_data(self, param):

        return

    def computeA(self, param, path):

        return

    def get_model(self, param, path):

        models = ['m1', 'm2', 'm3', 'm4']

        return models

    def computeB(self, param, path, model):

        return
