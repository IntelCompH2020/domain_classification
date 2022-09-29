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


class CorpusClassifierAL(object):
    """
    A container of corpus classification methods
    """

    def __init__(self, df_dataset, path2transformers=".", use_cuda=True):
        self.path2transformers = pathlib.Path(path2transformers)
        self.model = None
        self.df_dataset = df_dataset

    def load_model_config(self):
        path2model_config = self.path2transformers / "config.json"
        if not path2model_config.exists():
            model = ClassificationModel("roberta", "roberta-base", use_cuda=False)
            config = copy.deepcopy(model.config)
            # Save config
            config.to_json_file(path2model_config)
            del model
        else:
            config = RobertaConfig.from_json_file(path2model_config)
        self.config = config

    # def set_train_test(self, indices, isTrain=True, max_oversampling=10**100,nmax=10**100):

    #     iPositiveCount = self.df_dataset.loc[indices]['labels'].sum()
    #     iNegativeCount = len(indices)-iPositiveCount

    #     iMaxCount = np.min([np.max([iPositiveCount,iNegativeCount]),iPositiveCount*max_oversampling,iNegativeCount*max_oversampling])

    #     condition_positive = self.df_dataset.loc[indices]['labels']==1
    #     condition_negative = self.df_dataset.loc[indices]['labels']==0
    #     idx_positive = self.df_dataset.loc[indices][condition_positive].sample(iMaxCount, replace = True).index 
    #     idx_negative = self.df_dataset.loc[indices][condition_negative].sample(iMaxCount, replace = True).index

    #     indices = np.hstack([idx_positive[:int(nmax/2)],idx_negative[:int(nmax/2)]])

    #     assert len(indices) > 0
  
    #     if isTrain:
    #         self.indices_train = indices
    #     else:
    #         self.indices_test = indices
    #     return indices
    def set_indices_for_pu_learning(self, indices_train, indices_test=[]):
        self.indices_train = indices_train
        if len(indices_test) > 0:
            self.indices_test = indices_test


    def train_model(self, epochs=3, bUseWeakLabel=False, evaluate=True):

        logging.info("-- Training model...")

        # import pdb
        # pdb.set_trace()
        self.path2transformers.mkdir(exist_ok=True)
        model_dir = self.path2transformers / "best_model.pt"

        
        
        df_train = self.df_dataset.loc[self.indices_train]
        df_train['sample_weight'] = 1.
        if bUseWeakLabel:
            df_train.loc[:,['labels']] = df_train['weak_label'].copy()
        else:
            df_train.loc[:,['labels']] = df_train['trueAnnotation'].copy()


        # import pdb
        # pdb.set_trace()
        df_test = self.df_dataset.loc[self.indices_test]
        df_test['sample_weight'] = 1.


        # Load config
        self.load_model_config()

        self.model = CustomModel(self.config, self.path2transformers) if self.model == None else self.model
        # # FIXME: Base model 'roberta' should be configurable. Move it to
        # #        the config file (parameters.default.yaml)

        # Best model selection
        best_epoch = 0
        best_result = 0
        best_predictions = None
        best_model = None

        # Train the model
        logging.info(f"-- Training model with {len(df_train)} documents...")
        t0 = time()
        for e in tqdm(range(epochs), desc="Train epoch"):
            assert len(df_train) > 0
            epoch_loss, epoch_time = self.model.train_model(df_train)
            if evaluate:
                # Evaluate the model
                predictions, total_loss, result = self.model.eval_model(
                    df_test)

                #TAH:if result["f1"] > best_result:
                if result["f1"] >= best_result:
                    best_epoch = e
                    best_result = result["f1"]
                    best_predictions = predictions
                    best_model = copy.deepcopy(self.model)

        logging.info(f"-- -- Model trained in {time() - t0:.3f} seconds")

        if evaluate:
            self.model = best_model
            logging.info(f"-- Best model in epoch {best_epoch} with "
                         f"F1: {best_result:.3f}")

            delta = predictions[:, 1] - predictions[:, 0]
            df_test['prob_pred'] = 1 / (1 + np.exp(-delta))

        result = {'result':result, 
                'trainId': df_train['id'].to_numpy(),
                'pred_proba': df_test['prob_pred'].to_numpy(), 
                }

        return result

 
