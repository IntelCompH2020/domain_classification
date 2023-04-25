"""
PandasModifier encapsulate the logic to do the inference

@author: T.Ahlers
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import platform
import time
import pandas as pd

import torch.utils.data as data
from .custom_model_mlp import CustomDatasetMLP


class PandasModifier():

    def __init__(self, dh, kwargs):
        self.classifier = kwargs['classifier']
        self.dh = dh

    def start(self):
        pass

    def change(self, df):
        """
        does the inference and adds the soft and hard predictions to the
        dataframe
        Parameters
        ----------
        df: pd.DataFrame
        Returns
        -------
        df: pd.DataFrame
        """
        df_eval = df[['embeddings']].copy()
        df_eval.insert(0, 'labels', 0)
        eval_data = CustomDatasetMLP(df_eval)
        eval_iterator = data.DataLoader(eval_data, shuffle=False, batch_size=8)
        predictions = []
        for (x, y) in tqdm(eval_iterator, desc="Inference", leave=False):
            predictions_new = self.classifier(
                x).detach().cpu().numpy().reshape(-1)
            if len(predictions) == 0:
                predictions = predictions_new
            else:
                predictions = np.concatenate([predictions, predictions_new])

        df_prediction = pd.DataFrame(
            {'id': df['id'], 'prediction': (predictions > 0.5) * 1,
             'soft_prediction': predictions})
        return df_prediction


"""
Datahandler encapsulate logic to sequentially read all pandas file in a folder,
change them with the help of a PandasChanger object and write them to a
destination folder

@author: T.Ahlers
"""


class DataHandler():

    def __init__(self, paths: {}, config: {} = {}, **kwargs: {}) -> None:

        self.paths = paths
        self.lastFileName = ''
        self.folders = {}
        self.kwargs = kwargs
        self.__buildConfig(config, init=True)

    def run(self, sourceKey: str, destination_key: str, PandasChanger) -> None:
        """
        all files of a specific folder (given by source_key) gets changed with
        the given pandasChanger object and is written to the distiantion folder
        (given by destination_key)

        Parameters
        ----------
        sourceKey: encapsulate the source folder
        destination_key: encapsulate the destination folder
        PandasChanger: the object which encpasulate the logic how to change the
        pandas file

        Returns
        -------
        file_paths: str array
        """

        self.__isValidPath(sourceKey)
        self.__isValidPath(destination_key)

        pandasChanger = PandasChanger(self, self.kwargs)
        pandasChanger.start()

        while len(df := self.readNextFile(sourceKey)) > 0:
            df = pandasChanger.change(df)
            if len(df) == 0:
                continue

            # print('write')
            self.writeFile(destination_key, df)

    def refresh(self, init: bool = False, create: bool = False) -> None:
        """
        refresh the attribute self.folders based on the paths.

        Parameters
        ----------
        init: obsolete
        create: if True the folders of the path a created on the server
        """
        self.folders = {}
        fileSets = self.paths  # if init else self.folders
        for k, v in fileSets.items():
            folderPath = Path(v)  # if init else v
            # print(folderPath)
            k = self.__isValidDoubleKey(k)[0]
            if create:
                folderPath.mkdir(parents=True, exist_ok=True)
            self.folders[k] = {
                'folderPath': folderPath,
                'filePaths': np.array(
                    [os.path.join(r, n) for r, d, f in os.walk(folderPath)
                     for n in f if self.__isValidFile(n)]),
                'fileIdx': 0,
                'params': {}}

    def getFolder(self, key: str, **kwargs: {}) -> Path:
        """
        get path of folder

        Parameters
        ----------
        key: specifies the path

        Returns
        -------
        folder_path: str
        """
        self.__isValidPath(key)
        self.__buildConfig(kwargs)
        return self.folders[key]['folderPath']

    def getFiles(self, key: str, **kwargs: {}) -> []:
        """
        get path of files
        Parameters
        ----------
        key: specifies the path

        Returns
        -------
        file_paths: str array
        """
        self.__isValidPath(key)
        self.__buildConfig(kwargs)
        return self.folders[key]['filePaths']

    def readNextFile(self, key: str, **kwargs: {}) -> []:
        """
        reads the next file in the sequence
        Parameters
        ----------
        key: specifies the path

        Returns
        -------
        df: pd.DataFrame of file content
        """
        self.__isValidPath(key)
        self.__buildConfig(kwargs)

        if self.folders[key]['fileIdx'] >= len(self.folders[key]['filePaths']):
            return []

        fileName = self.folders[key]['filePaths'][self.folders[key]['fileIdx']]
        if self.config['fileType'] != '':
            if fileName.split('.')[-1] != self.config['fileType']:
                self.folders[key]['fileIdx'] += 1
                # print(f'recursion: {fileName}')
                return self.readNextFile(key)

        self.lastFileName = fileName
        # print(self.lastFileName)
        result = pd.read_parquet(self.lastFileName)
        self.folders[key]['fileIdx'] += 1
        return result

    def writeFile(self, key: str, data: object, fileName: str = '') -> None:
        """
        writes a new parquet file
        Parameters
        ----------
        key: specifies the path
        data: the file content
        fileName: name of file
        """

        fileName = (fileName if fileName != ''
                    else self.lastFileName.split('/')[-1])
        self.__isValidPath(key)
        if self.lastFileName == '':
            # print('da')
            raise Exception('last file name is missing')
        # print(f'write:{self.folders[key]["folderPath"]/fileName}')
        data.to_parquet(self.folders[key]['folderPath'] / fileName)
        self.lastFileName = ''

    def __buildConfig(self, config, init=False):
        """
        builds the config dictionary by mixing the given config with the
        passed config

        Parameters
        ----------
        config: config dictionary
        """
        dConfig = {'fileType': 'parquet',
                   'minAge': 0,
                   'debug': False,
                   'create': True}
        self.config = {}
        change = False
        for k, v in dConfig.items():
            if k not in config:
                self.config[k] = dConfig[k]
            else:
                change = True
                self.config[k] = config[k]

        if init or change:
            self.refresh(init, self.config['create'])

    def __createDeltaFiles(self, keys: str) -> None:
        """
        create the delta between the destination folder and source folder so
        that just the delta gets proceed

        Parameters
        ----------
        key: specifies the path
        """

        keyA, keyB = self.__isValidDoubleKey(keys)
        self.__isValidPath(keyA, tryFix=False)
        self.__isValidPath(keyB, tryFix=False)
        minAge = self.config['minAge']
        filesA = [np.array(f.split('/'))[-1]
                  for f in self.folders[keyA]['filePaths']]
        filesB = [np.array(f.split('/'))[-1]
                  for f in self.folders[keyB]['filePaths']]

        deltaMask = (np.in1d(filesA, filesB) is False)

        keyBigSet = (
            keyA if (len(self.folders[keyA]['filePaths'])
                     >= len(self.folders[keyB]['filePaths'])) else keyB)

        filesBigSet = (
            self.folders[keyBigSet]['filePaths'] if minAge == 0
            else self.__getFilesFiltered(
                keyBigSet, self.folders[keyBigSet]['filePaths'], minAge))

        keySmallSet = (
            keyA if (len(self.folders[keyA]['filePaths'])
                     < len(self.folders[keyB]['filePaths'])) else keyB)

        filesNamesBigSet = [np.array(f.split('/'))[-1] for f in filesBigSet]

        filesNamesSmallSet = [np.array(f.split('/'))[-1]
                              for f in self.folders[keySmallSet]['filePaths']]

        deltaMask = (np.in1d(filesNamesBigSet, filesNamesSmallSet) is False)

        self.folders[keys] = {
            'folderPath': self.folders[keyBigSet]['folderPath'],
            'filePaths': filesBigSet[deltaMask],
            'fileIdx': 0,
            'params': {'minAge': minAge}}

    def __getFilesFiltered(self, key: str, fileNames: [], minAge: int) -> []:
        """
        return just the files which are older than than the minimum age
        Parameters
        ----------
        key: specifies the path
        fileNames: list of files
        minAge: minimum age of file
        Returns
        -------
        file_list: list of str

        """
        returnFileNames = []
        for fileName in fileNames:
            if (time.time() - self.__get_file_creation_date(
                    self.folders[key]['folderPath'] / fileName)) < minAge:
                continue
            returnFileNames.append(fileName)
        return np.array(returnFileNames)

    def __get_file_creation_date(self, path_to_file) -> float:
        """
        returns the file creation date
        Parameters
        ----------
        path_to_file: file path
        Returns
        -------
        timestamp:
        """

        if platform.system() == 'Windows':
            return os.path.getctime(path_to_file)
        try:
            stat = os.stat(path_to_file)
            return stat.st_birthtime
        except AttributeError:
            return stat.st_mtime

    def __isValidFile(self, filePath: str):
        """
        returns if the file type is correct
        Parameters
        ----------
        file_path: file path
        Return
        ----------
        is_file_correct: bool

        """
        if self.config['fileType'] == '':
            return True
        if self.config['fileType'] == filePath.split('.')[-1]:
            return True
        return False

    def __isValidPath(self, key: str, tryFix: bool = True) -> None:
        """
        raises an exception if the path is not valid
        Parameters
        ----------
        key: file path
        Exceptions
        ----------
        ex: encapsulate that the path is wrong
        """
        ex = Exception("path is not registered")
        if key not in self.folders.keys():
            if tryFix:
                try:
                    self.__createDeltaFiles(key)
                except Exception:
                    raise ex
            else:
                raise ex

    def __isValidDoubleKey(self, initKey: str) -> None:
        """
        returns if a it is a valid double key
        ----------
        initKey: str double key
        Returns
        -------
        keyParts: parts of the key in a list
        Exception
        -------
        Exception: encapsulate that the given format is wrong

        """
        keyParts = initKey.split('_')
        if len(keyParts) != 2:
            raise Exception('format of key has to be x_x')
        return keyParts
