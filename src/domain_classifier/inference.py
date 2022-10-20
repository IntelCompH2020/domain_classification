"""
PandasModifier encapsulate logic to do the inference
DataHandler is a helper class for data handling 

@author: T.Ahlers
"""

import numpy as np

from tqdm import tqdm

from transformers.models.mpnet.configuration_mpnet import MPNetConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data as Dataset
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import yaml
from pathlib import Path
import os

import os
from pathlib import Path
import numpy as np
import pandas as pd

from .custom_model_mlp import MLP
from .custom_model_mlp import CustomDatasetMLP

class PandasModifier(): 
	def __init__(self,dh,kwargs):
		self.classifier = kwargs['classifier']
		self.dh = dh
	def start(self):
		pass
	def change(self,df):
		df_eval = df[['embeddings']].copy()
		df_eval.insert(0,'labels',0)
		eval_data = CustomDatasetMLP(df_eval)
		eval_iterator = data.DataLoader(eval_data,shuffle=False,batch_size=8)
		predictions = []
		for (x, y) in tqdm(eval_iterator, desc="Inference", leave=False):
			predictions_new = self.classifier(x).detach().cpu().numpy().reshape(-1)
			if len(predictions) == 0:
				predictions = predictions_new
			else:
				predictions = np.concatenate([predictions,predictions_new])
		df_prediction = pd.DataFrame({'id':df['id'],'prediction':(predictions>0.5)*1,'soft_prediction':predictions})
		return df_prediction

class DataHandler():
	def __init__(self,paths: {}, config: {} = {}, **kwargs: {}) -> None: 
		self.paths = paths 
		self.lastFileName = ''
		self.folders = {}
		self.kwargs = kwargs
		self.__buildConfig(config,init=True)
	def __buildConfig(self,config,init=False):
		dConfig = { 'fileType': 'parquet',
					'minAge': 0,
					'debug': False,
					'create': True }
		self.config = {}
		change = False
		for k,v in dConfig.items():
			if k not in config:
				self.config[k] = dConfig[k]
			else:
				change = True
				self.config[k] = config[k] 
		  
		if init or change:
			self.refresh(init,self.config['create'])    
    #bug for not init
	def refresh(self,init:bool = False, create: bool = False) -> None: 
		self.folders = {}
		fileSets = self.paths #if init else self.folders 
		for k,v in fileSets.items():
			folderPath = Path(v) #if init else v
			#print(folderPath)
			k = self.__isValidDoubleKey(k)[0]
			if create:
				folderPath.mkdir(parents=True, exist_ok=True)
			self.folders[k] = { 'folderPath': folderPath,
								'filePaths': np.array([os.path.join(r,n) for r,d,f in os.walk(folderPath) for n in f if self.__isValidFile(n)]),
								'fileIdx': 0,
								'params': {} }

	def getFolder(self,key:str, **kwargs: {}) -> Path:
		self.__isValidPath(key)
		self.__buildConfig(kwargs)
		return self.folders[key]['folderPath']   
	def getFiles(self,key:str, **kwargs: {}) -> []:
		self.__isValidPath(key)
		self.__buildConfig(kwargs)
		return self.folders[key]['filePaths'] 
	def readNextFile(self,key:str, **kwargs: {})-> []:
		self.__isValidPath(key)
		self.__buildConfig(kwargs)
		if self.folders[key]['fileIdx'] >= len(self.folders[key]['filePaths']): return []
		fileName = self.folders[key]['filePaths'][self.folders[key]['fileIdx']]
		if self.config['fileType'] != '':
		    if fileName.split('.')[-1] != self.config['fileType']:
		        self.folders[key]['fileIdx'] += 1
		        #print(f'recursion: {fileName}')
		        return self.readNextFile(key)
		self.lastFileName = fileName
		#print(self.lastFileName)
		result = pd.read_parquet(self.lastFileName)
		self.folders[key]['fileIdx'] += 1
		return result
	def writeFile(self,key:str,data: object, fileName:str = '')-> None:
		fileName = fileName if fileName != '' else self.lastFileName.split('/')[-1]
		self.__isValidPath(key)
		if self.lastFileName == '':
		    #print('da')
		    raise Exception('last file name is missing')
		#print(f'write:{self.folders[key]["folderPath"]/fileName}')
		data.to_parquet(self.folders[key]['folderPath']/fileName)
		self.lastFileName = ''
	def run(self,sourceKey:str,destination_key: str,PandasChanger)-> None:
		self.__isValidPath(sourceKey)
		self.__isValidPath(destination_key)
		pandasChanger = PandasChanger(self,self.kwargs)
		pandasChanger.start()
		while len(df := self.readNextFile(sourceKey))>0:
		    df = pandasChanger.change(df)
		    if len(df) == 0: continue
		    #print('write')
		    self.writeFile(destination_key,df)
	def test(self):
		return self.folders

	def __createDeltaFiles(self,keys: str) -> None:
	    keyA,keyB = self.__isValidDoubleKey(keys)
	    self.__isValidPath(keyA,tryFix=False)
	    self.__isValidPath(keyB,tryFix=False)
	    minAge = self.config['minAge']
	    filesA = [np.array(f.split('/'))[-1] for f in self.folders[keyA]['filePaths']]
	    filesB = [np.array(f.split('/'))[-1] for f in self.folders[keyB]['filePaths']]
	    
	    deltaMask = (np.in1d(filesA,filesB) == False)
	    
	    keyBigSet = keyA if len(self.folders[keyA]['filePaths']) >= len(self.folders[keyB]['filePaths']) else keyB 
	    filesBigSet = self.folders[keyBigSet]['filePaths'] if minAge == 0 else self.__getFilesFiltered(keyBigSet,self.folders[keyBigSet]['filePaths'],minAge)
	    keySmallSet = keyA if len(self.folders[keyA]['filePaths']) < len(self.folders[keyB]['filePaths']) else keyB 
	    filesNamesBigSet = [np.array(f.split('/'))[-1] for f in filesBigSet]
	    filesNamesSmallSet = [np.array(f.split('/'))[-1] for f in self.folders[keySmallSet]['filePaths']]
	    deltaMask = (np.in1d(filesNamesBigSet,filesNamesSmallSet) == False)
	    
	    self.folders[keys] = {  'folderPath': self.folders[keyBigSet]['folderPath'],
	                            'filePaths': filesBigSet[deltaMask],
	                            'fileIdx': 0,
	                            'params': { 'minAge': minAge } }
	def __getFilesFiltered(self,key: str, fileNames: [], minAge: int )-> []:
	    returnFileNames = []
	    for fileName in fileNames:
	        if (time.time() - self.__get_file_creation_date(self.folders[key]['folderPath'] / fileName)) < minAge:
	            continue
	        returnFileNames.append(fileName)
	    return np.array(returnFileNames)
	def __get_file_creation_date(self,path_to_file)-> float: 
	    if platform.system() == 'Windows':
	        return os.path.getctime(path_to_file)
	    try:
	        stat = os.stat(path_to_file)
	        return stat.st_birthtime
	    except AttributeError:
	        return stat.st_mtime
	def __isValidFile(self, filePath:str):
	    if self.config['fileType'] == '': return True
	    if self.config['fileType'] == filePath.split('.')[-1]: return True
	    return False
	def __isValidPath(self,key:str,tryFix: bool = True)-> None: 
	    ex = Exception("path is not registered")
	    if key not in self.folders.keys():
	        if tryFix:
	            try:
	                self.__createDeltaFiles(key)
	            except:
	                raise ex
	        else:
	            raise ex
	def __isValidDoubleKey(self,initKey: str)-> None: 
	    keyParts = initKey.split('_')
	    if len(keyParts) != 2:
	        raise Exception('format of key has to be x_x')
	    return keyParts