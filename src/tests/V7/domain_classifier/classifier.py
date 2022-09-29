#simple transformer

import logging
import pathlib
import sys
from time import time
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import copy
from simpletransformers.classification import ClassificationModel
from transformers.models.roberta.configuration_roberta import RobertaConfig

from sklearn import model_selection
from tqdm import tqdm

from transformers import logging as hf_logging
import copy

try:
    from .custom_model import CustomModel
except:
    from custom_model import CustomModel

# Mnemonics for values in column 'train_test'
TRAIN = 0
TEST = 1
# Equivalent to NaN for the integer columns in self.df_dataset:
# (nan is not used because it converts the whole column to float)
UNUSED = -99

hf_logging.set_verbosity_error()

#early stopping

class CorpusClassifier(object):
    """
    A container of corpus classification methods
    """

    def __init__(self, path2transformers="."):
        self.path2transformers = pathlib.Path(path2transformers)
        self.model = None

    def load_model_config(self):
        path2model_config = self.path2transformers / "config.json"
        if not path2model_config.exists():
            #model = ClassificationModel("roberta", "roberta-base", use_cuda=False)
            model = ClassificationModel("roberta", "roberta-base", num_labels =2, weight = [1,1], use_cuda=False)
            config = copy.deepcopy(model.config)
            # Save config
            config.to_json_file(path2model_config)
            del model
        else:
            config = RobertaConfig.from_json_file(path2model_config)
        self.config = config

#epochs=-1 => early stopping
    def train_loop(self, df_train, df_eval=[], epochs=-1):

        is_early_stopping = (epochs < 0)
        if is_early_stopping:
            epochs = 5

        #add sample weight for custom_model
        df_train.loc[:,['sample_weight']] = 1.
        if len(df_eval) > 0:
            df_eval.loc[:,['sample_weight']] = 1.


        self.path2transformers.mkdir(exist_ok=True)
        model_dir = self.path2transformers / "best_model.pt"
        self.load_model_config()
        self.model = CustomModel(self.config, self.path2transformers) if self.model == None else self.model

        result = {}

        #for e in tqdm(range(epochs), desc="Train epoch"):
        epoch = 0
        loss_history = []
        bStop = False
        f1_cont = -1
        while True:
            for mode in ['train','eval']:
                if mode == 'train':
                    epoch_loss, epoch_time = self.model.train_model(df_train) 
                    loss_history.append({ 'train_loss': epoch_loss }) 
                elif len(df_eval) > 0:
                    predictions, total_loss, result = self.model.eval_model(df_eval)    
                    
                    loss_history[-1] = {**loss_history[-1], **{ 'eval_loss': total_loss }, ** result }

                    if 1 == 1:
                 
                        pHistory = {k:v for k,v in loss_history[-1].items() if ('cont' in k) or ('loss' in k) }
                        print(f'evaluation: { pHistory }')

                    #no early stopping => no break up condition
                    if is_early_stopping == False:
                        last_history = loss_history[-1]
                        continue
                    #first epoch => nothing to compare => no break up condition
                    if len(loss_history) == 1:
                        last_history = loss_history[-1]
                        last_model = copy.deepcopy(self.model)
                        continue
                    #current loss smaller than previous loss => continue

                    
                    if loss_history[-1]['f1_cont'] > (loss_history[-2]['f1_cont'] + 0.001):
                        if 1 == 1:
                            f1_current = loss_history[-1]['f1_cont']
                            f1_old = loss_history[-2]['f1_cont']
                            print(f'Current F1 score ({f1_current}) > then old f1 score ({f1_old})=>continue' )
                        last_history = loss_history[-1]
                        last_model = copy.deepcopy(self.model)
                        continue

                    #otherwise stop it
                    bStop = True
                    self.model = last_model
                    if 1 == 1:
                        f1_current = loss_history[-1]
                        f1_old = loss_history[-2]['f1_cont']
                        print(f'Current F1 score ({f1_current}) is not ( or just slighty ) better than old f1 score ({f1_old})=>stop' )
                        
                    break
                else:
                    if is_early_stopping:
                        break #no test data 

            if bStop:
                break
            epoch +=1
            if is_early_stopping == False and epoch == epochs:
                break   

        return last_history  


    # delta = predictions[:, 1] - predictions[:, 0]
    # df_test['prob_pred'] = 1 / (1 + np.exp(-delta))
    # result = {'result':result, 
    #           'pred_proba': df_test['prob_pred'].to_numpy(), 
    #          }


    def eval(self, df_eval):
        df_eval.loc[:,['sample_weight']] = 1.

        predictions, total_loss, result = self.model.eval_model(df_eval)
        
        from scipy.special import softmax
        predictions = softmax(predictions, axis=1)

        #print(f'clf.eval: {result}')
        return result
        #return [predictions[:,1], result]

    # def eval2(self, df_eval):
    #     df_eval.loc[:,['sample_weight']] = 1.

    #     predictions, total_loss, result = self.model.eval_model(df_eval)
    #     delta = predictions[:, 1] - predictions[:, 0]

    #     result = {'result':result, 
    #               'pred_proba': (1 / (1 + np.exp(-delta))), 
    #              }

    #     return result

    def save(self,model_name='last_model.pt'):
        model_dir = f'{ self.path2transformers }/{ model_name }'

        #self.model.freeze_encoder_layer()

        self.model.save(model_dir)

    def load(self,model_name='last_model.pt'):
        model_dir = f'{ self.path2transformers }/{ model_name }'
        self.load_model_config()
        self.model = CustomModel(self.config, self.path2transformers)
        self.model.load(model_dir)

    def predict(self, df_eval):
        df_eval.loc[:,['sample_weight']] = 1.
        predictions = self.model.predict(df_eval)

        result = (np.array(predictions[:, 0] ) < np.array(predictions[:, 1]))*1

        return result

    def predict_proba(self, df_eval):
        df_eval.loc[:,['sample_weight']] = 1.
        predictions = self.model.predict(df_eval)

        from scipy.special import softmax
        result = softmax(predictions, axis=1)

        return result[:,1] #probability class 1

    # def predict_proba(self, df_eval):
    #     df_eval.loc[:,['sample_weight']] = 1.

    #     predictions, total_loss, result = self.model.eval_model(df_eval)
    #     delta = predictions[:, 1] - predictions[:, 0]

    #     result = {'result':result, 
    #               'pred_proba': str((1 / (1 + np.exp(-delta)))), 
    #              }

    #     return result


