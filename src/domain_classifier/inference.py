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

class CustomDataset(Dataset):
	def __init__(self,df_data):
		self.data = df_data[['embeddings','labels']].to_numpy().copy()
		#print(f'self.data:{self.data}')
	def __getitem__(self,idx):
		item = torch.Tensor(self.data[idx,0]).to('cuda'), torch.Tensor([self.data[idx,1]]).to('cuda')
		#print(f'idx:{idx}')
		return item
	def __len__(self):
		return len(self.data)
    
class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
	    super().__init__()
	    self.linear1 = nn.Linear(input_dim, hidden_dim)
	    self.linear2 = nn.Linear(hidden_dim, output_dim)
	    self.criterion = nn.BCELoss()
	    self.optimizer = optim.Adam(self.parameters(),lr=0.001)
	    self.to('cuda')
	def forward(self, x):
	    #print(f'x1:{x}')
	    x = self.linear1(x)
	    #print(f'x2:{x}')
	    x = F.relu(x)
	    x = F.sigmoid(self.linear2(x))
	    return x
	def save(self):
		pass
	def calculate_f1_score(self,y_preds,y_labels):
	    epsilon = 1e-7
	    y_preds = y_preds.view(-1)
	    y_labels = y_labels.view(-1)

	    tp_c = (y_preds * (y_labels == 1)).sum()
	    fp_c = (y_preds * (y_labels == 0)).sum()
	    tn_c = ((1-y_preds) * (y_labels == 1)).sum()
	    fn_c = ((1-y_preds) * (y_labels == 0)).sum()
	    
	    print(f'tp_c/fp_c/tn_c/tp_c:{tp_c}/{fp_c}/{tn_c}/{fn_c}')

	    precision = tp_c/(tp_c+fp_c+epsilon)
	    recall = tp_c/(tp_c+fn_c+epsilon)
	    return 2 * precision*recall/(precision+recall)

	def predict_proba(self,df_eval):
		eval_data = CustomDataset(df_eval)
		eval_iterator = data.DataLoader(eval_data,shuffle=False,batch_size=8)
		predictions = []
		for (x, y) in tqdm(eval_iterator, desc="Inference", leave=False):
			predictions_new = self.forward(x).detach().cpu().numpy()
			if len(predictions) == 0:
				predictions = predictions_new
			else:
				predictions = np.concatenate([predictions,predictions_new])
		        
		return predictions.reshape(-1)
	        
	        
	def train_loop(self,train_iterator,eval_iterator, epochs=-1, device='cuda'):
		train_loss = 0
		eval_loss = 0
		eval_losses = []
		train_losses = []
		early_stopping = epochs == -1
		y_labels,y_preds,f1_scores = [],[],[]
		while True:
			self.train()
		    
			for (x, y) in tqdm(train_iterator, desc="Training", leave=False):
				#∫print(f'loopX/Y:{x}/{y}')
				self.optimizer.zero_grad()
				y_pred = self.forward(x)
				loss = self.criterion(y_pred, y)
				loss.backward()
				self.optimizer.step()
				train_loss += loss.item()
			    
			self.eval()
			with torch.no_grad():
				for (x, y) in tqdm(eval_iterator, desc="Eval", leave=False):
					y_pred = self.forward(x)
					loss = self.criterion(y_pred, y)
					eval_loss += loss.item()
					if len(y_preds) == 0:
						y_preds = y_pred
						y_labels = y
					else:
						y_preds = torch.concat([y_preds,y_pred])
						y_labels = torch.concat([y_labels,y])
			        
			f1_scores.append(self.calculate_f1_score(y_preds,y_labels).detach().cpu().numpy())
			y_labels,y_preds = [],[]

			train_losses.append(train_loss/len(train_iterator)) 
			eval_losses.append(eval_loss/len(eval_iterator))

			print(f'Train/eval loss/eval_f1_score:{train_losses[-1]}/{eval_losses[-1]}/{f1_scores[-1]}')

			if len(train_losses) == epochs:
			    break

			if early_stopping:
				metrics = -np.array(f1_scores) #eval_scores
				#metrics = np.array(eval_losses)
				metrics[-3:] *= 0.99 #to forece a minimum improvement
				if np.argmin(metrics) + 1 == len(metrics):
					best_model = self
				    
				if np.argmin(metrics) + 3 <= len(metrics):
					self = best_model
					break   
		return [train_losses[-1],eval_losses[-1],f1_scores[-1]]

class Inference(object):
	def __init__(self,global_parameters):
		#self.global_parameters = global_parameters
		self.state = { 'model': None, 'dataset': None }
		self.modelOptions = global_parameters['models']
		self.datasetsOptions = global_parameters['datasets']
		self.printModelOptions = [ f'Select model {r["name"]}' for r in global_parameters['models']]
		self.printDatasetsOptions = [ f'Select dataset {r["name"]}' for r in global_parameters['datasets']]
		self.pm = ProjectManager('MyProject2')
	def getOptions(self):
		if self.state['model'] == None:
			return self.printModelOptions
		if self.state['dataset'] == None:
			return self.printDatasetsOptions
		return [f'Infer dataset { self.state["dataset"]["name"] } using model { self.state["model"]["name"] }']
	def setOption(self,option):
		isModel = np.sum(np.array(self.printModelOptions) == option) > 0
		isDataset = np.sum(np.array(self.printDatasetsOptions) == option) > 0
		option = ' '.join(np.array(option.split(' '))[2:])

		if isModel:
			dOption = self.modelOptions[np.nonzero(np.array([r['name'] for r in self.modelOptions]) == option)[0][0]]
			self.state['model'] = dOption
		elif isDataset:
			dOption = self.datasetsOptions[np.nonzero(np.array([r['name'] for r in self.datasetsOptions]) == option)[0][0]]
			self.state['dataset'] = dOption
		else:
			self.__inferData()
	def __inferData(self):
		classifier = MLP(768,1024,1)
		classifier.load_state_dict(torch.load(self.pm.getModelPath(self.state['model']['filename'])))
		classifier.eval()
		dPaths = {  'd_documentEmbeddings': self.state['dataset']['path'],
            		'p_prediction': f'/export/data_ml4ds/IntelComp/Code/Tool/domain_classification/MyProject2/output/{self.state["dataset"]["name"]}' 
        }
		dh = DataHandler(dPaths, classifier = classifier)
		dh.run('p_d','p',PandasModifier)


from pathlib import Path
import os
class ProjectManager(object):
	def __init__(self,projectName):
		self.name = projectName
	def getModelPath(self,model_name):
		return Path(f'/export/data_ml4ds/IntelComp/Code/Tool/domain_classification/MyProject2/models/{model_name}')

import os
from pathlib import Path
import numpy as np
import pandas as pd


class PandasModifier(): 
	def __init__(self,dh,kwargs):
		self.classifier = kwargs['classifier']
		self.dh = dh
	def start(self):
		pass
	def change(self,df):
		df_eval = df[['embeddings']].copy()
		df_eval.insert(0,'labels',0)
		eval_data = CustomDataset(df_eval)
		eval_iterator = data.DataLoader(eval_data,shuffle=False,batch_size=8)
		predictions = []
		for (x, y) in tqdm(eval_iterator, desc="Inference", leave=False):
			predictions_new = self.classifier.forward(x).detach().cpu().numpy().reshape(-1)
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




