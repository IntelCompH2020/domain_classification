# try:
from .custom_model import CustomModel
from .query_strategy import WeakSoftLabelTrustSampling
import numpy as np
from copy import copy, deepcopy
from sklearn.model_selection import train_test_split
import pandas as pd

import time
import os
from pathlib import Path
import json


class ActiveLearner(object):
    def __init__(self, clf, queryStrategy, df_dataset=[], dConfig = None):
        self.queryStrategy = queryStrategy
        self.dConfig = self._getDefaultConfig()
        if dConfig != None:
            self._overwriteDefaultConfig(dConfig)

        if len(df_dataset) > 0:
            self.df_dataset = df_dataset
        else:
            self.df_dataset = self.load()

        self.df_dataset.loc[:,['labels']] = -1
        self.df_dataset.loc[:,['weak_label']] = -1
        self.df_dataset.loc[:,['is_validation']] = False
        self.df_dataset = self.df_dataset.sort_values(by=['weak_soft_label'], ascending = False)

        self.clf = clf
        self.clfs = []

        self.indices_queried = []
        self.protocol = []

    def _getDefaultConfig(self):
        return { 'path': 'data/activeLearning/',
                 'test_size': 0.5, #test size for df_true label split
                 'weightFactor': 1, #the heigher the factor the more relevant are clfs with higher f1 scores (for sampling)
                 #'init_weak_label_ratio': 0.1, #weak label ratio for the first base classifier
                 'backup_threshold': 0.2, #if both classes are available after first annotation example it is calculated by that, backup is this value
                 #'init_oversampling_rate': 4, #oversampling rate for the first base classifier
                 'quantile_95': 1, #influence the noise
                 'annotationRatioFactor': 0.2,
                 'printSteps': True}  #if true printouts are done during execution so that it is easier to understand #1 max change: 2x, min change = x/2

    def _overwriteDefaultConfig(self,dConfig):
        for key,value in dConfig.items():
            if key in self.dConfig:
                self.dConfig[key] = value      
        

    def _getConfigParameter(self,name):
        return self.dConfig[name]

    def _printIt(self,text):
        if self._getConfigParameter('printSteps'):
            print(text)

          
    def query(self, num_samples=10):

        # if len(self.clfs) > 0:
        #     dClf = self._getClf('is_query_lead')

        clf = None if len(self.clfs) == 0 else self._getClf()['clf']
        
        df_unlabeled = self._get_sub_set('unlabeled')

        self.indices_queried, predict_proba = self.queryStrategy.query(clf = clf, 
                                                                       df_dataset = df_unlabeled,
                                                                       n = num_samples, #self._getRequestedClasses(num_samples),
                                                                       weak_soft_label_trust = 1 )

        self._printIt(f'Queried new indices{self.indices_queried} with proba {predict_proba}')

        return self.indices_queried
    
    def update(self, y, bSave=True):

###########################UPDATE DATASET###########################

        #UPDATE DATASET
        self.df_dataset.loc[self.indices_queried,['labels']] = y
        self.df_dataset.loc[self.indices_queried,['annotation_idx']] = (np.max(self.df_dataset['annotation_idx'].to_numpy()) + 1)

        #UPDATE CLASSIFIER
        if len(self.clfs) == 0:
            weak_label_threshold = self._get_weak_label_threshold()
            clf = deepcopy(self.clf)
            clf.load('base_model.pt')
            self.clfs = [{ 'weak_label_threshold': weak_label_threshold, 'idx': 0, 'clf': clf, 'f1_cont': 0 }]
        else:
            clf = deepcopy(self.clfs[-1])
            weak_label_threshold = self._get_weak_label_threshold()
            self.clfs = [{ 'weak_label_threshold': weak_label_threshold, 'idx': 0, 'clf': clf, 'f1_cont': 0 }]

        clf = self.clfs[-1]['clf']

        #EXPLOIT TRUE ANNOTATIONS
        df_fresh_annotated = self._get_sub_set('fresh_annotated') 
        df_train_true1, df_test = train_test_split( df_fresh_annotated, test_size=0.5, random_state=42, stratify = df_fresh_annotated['labels'].to_numpy())
        self.df_dataset.loc[df_test.index,['is_validation']] = True

        df_train_true = self._get_sub_set('train_annotation')
        df_train_true,_ = self._oversample_minority_class(df_train_true)

        df_validation = self._get_sub_set('validation')
        df_validation = self._buildTestSet(df_validation,weak_label_threshold)

        dclf = self.clfs[-1]

        last_history = clf.train_loop(df_train_true,df_validation)
        dclf['f1_cont'] = last_history['f1_cont']
        dclf['last_history'] = last_history


# ###########################EVALUATE CLFS###########################
#         if len(self.clfs) > 0:
#             n_classifier = len(self.clfs)
#             #f1_conts = self._evaluateClfs() #true
#             #self._printIt(f'The f1_scores of the classifiers are {f1_conts}')
#             #self._createMainClfByF1Scores(f1_conts)
#             self._removeNoneBaseClassifier()
#             self._printIt(f'Base Classifier 0:{self.clfs[-1]}')          
#         else:
#             self._createMainClfInit()
#             self._printIt(f'Initial Base Classifier 0:{self.clfs[-1]}')
            
        
        

# ###########################create mutations###########################
#         for idx,mutationAttribute in enumerate(['weak_label_ratio','weak_label_threshold','oversampling_rate']):
#             self._mutateBaseClf(mutationAttribute)
#             self._printIt(f'Mutation Classifier {idx+1}:{self.clfs[-1]}')

# ###########################train all classifier###########################
#         for dClf in self.clfs:
#             df_train_true_label, df_test_true_label, df_weak_label = self._buildTrueAndWeakLabelDataset(dClf['weak_label_threshold'],dClf['oversampling_rate'])
#             dClf['df_test_true_label'] = df_test_true_label
#             self._train_classifier(df_train_true_label, df_test_true_label, df_weak_label,dClf)

#         self._determineQueryClf()

#returns either the base classifier (the classifier who won't be mutated)
#        or the query classifier when attribute ='is_query_lead'
    def _getClf(self,attribute='is_base'): #is_query_lead
        return self.clfs[-1]
        #return list(filter(lambda dclf: dclf[attribute], self.clfs))[0]

    def _createMainClfInit(self):
        weak_label_threshold = self._get_weak_label_threshold()
        weak_label_ratio = self._getConfigParameter('init_weak_label_ratio') 
        oversampling_rate = self._getConfigParameter('init_oversampling_rate') 
        clf = deepcopy(self.clf)
        self.clfs = [{ 'weak_label_ratio': weak_label_ratio, 'weak_label_threshold': weak_label_threshold, 'oversampling_rate': oversampling_rate, 'idx': 0, 'clf': clf, 'true_weak_test_idxs':[], 'f1_cont': 0, 'is_base': True, 'is_query_lead': False, 'df_test_true_label': [] }]

    def _createMainClfByF1Scores(self,f1_conts):
        idx_new_base_classifier = self._getDecisionByF1Scores(f1_conts)
        dClfMain = self.clfs[idx_new_base_classifier].copy()
        clf = deepcopy(self.clf)
        dClfMain = { **dClfMain, **{ 'clf': clf, 'is_base': True, 'is_query_lead': False, 'df_test_true_label': [], 'idx': 0, 'true_weak_test_idxs':[]  } } #overwrite attributes clf, is_base and query_lead
        self.clfs = [ dClfMain ]

    def _removeNoneBaseClassifier(self):
        dClfMain = self._getClf('is_query_lead')
        clf = deepcopy(self.clf)
        dClfMain = { **dClfMain, **{ 'clf': clf, 'is_base': True, 'is_query_lead': False, 'df_test_true_label': [], 'idx': 0, 'true_weak_test_idxs':[]  } }
        self.clfs = [ dClfMain ]

    def _mutateBaseClf(self,attribute):
        dMutationClf = self._getClf('is_base').copy()
        noiseFactor = self._getNoiseFactorByF1Score(dMutationClf['f1_cont'])
        if attribute == 'weak_label_threshold':
            positiveCount = len(self.df_dataset[self.df_dataset['weak_soft_label']>dMutationClf['weak_label_threshold']]) * noiseFactor #threshold is difficult to interpret hece we change the positiveCount accoridng to the noise factor
            positiveCount = int(np.min([positiveCount,len(self.df_dataset)-1]))  #cannot be higher than lenth of dataset
            dMutationClf[attribute] = self.df_dataset.sort_values(by = ['weak_soft_label'], ascending = False).iloc[positiveCount]['weak_soft_label'] #translate postivecount to threshold 

        elif attribute == 'weak_label_ratio': #cannot be higher than 1
            dMutationClf[attribute] *= noiseFactor #
            dMutationClf[attribute] = np.min([1,dMutationClf[attribute]])
        else: #oversampling_rate has to be at least 1
            dMutationClf[attribute] *= noiseFactor
            dMutationClf[attribute] = np.max([1,dMutationClf[attribute]])
           
        dMutationClf['idx'] = len(self.clfs) 
        self.clfs.append(dMutationClf)

    def _getDecisionByF1Scores(self,f1_conts):
        eps = 10**-10
        error = 1/(1-np.array(f1_conts)+eps)
        weightFactor = self._getConfigParameter('weightFactor')
        weights = np.power(error,weightFactor)
        weights = weights/np.sum(weights)
        cumsum_weights = np.cumsum(weights)

        random_number = np.random.uniform()
        idx_classifier = np.nonzero(random_number < cumsum_weights)[0][0]
        self._printIt(f'The f1 scores of the classifier are {f1_conts} and the weightFactor is { weightFactor } and the formula is 1/(1-np.array(f1_conts)+eps)')
        self._printIt(f'Classifier { idx_classifier } is query lead, because the cumulated weights are { cumsum_weights } and the random number is { random_number }')
        return idx_classifier


    def _determineQueryClf(self):
        f1_conts = [ dClf['f1_cont'] for dClf in self.clfs]
        idx_query_classifier = self._getDecisionByF1Scores(f1_conts)
        self.clfs[idx_query_classifier]['is_query_lead'] = True

    def _evaluateClfs(self):
        
        f1_conts = []
        for idx,dclf in enumerate(self.clfs):
            df_eval = pd.concat([dclf['df_test_true_label'], self._get_sub_set('fresh_annotated')])
            f1_conts.append(dclf['clf'].eval(df_eval)['f1_cont'])
            dclf['f1_cont'] = f1_conts[-1]
        return f1_conts

    def _get_condition(self,name):
        return { 'labeled': (self.df_dataset['labels'] != -1),
                 'unlabeled': (self.df_dataset['labels'] == -1),
                 'positive': (self.df_dataset['labels'] == 1),
                 'negative': (self.df_dataset['labels'] == 0),
                 'validation': (self.df_dataset['is_validation']),
                 'train_annotation': (self.df_dataset['is_validation'] == False) & (self.df_dataset['annotation_idx'] > -1),
                 'fresh_annotated': (self.df_dataset['annotation_idx'] == np.max(self.df_dataset['annotation_idx'].to_numpy()))}[name]
    def _get_sub_set(self,name):
        return self.df_dataset[self._get_condition(name)]

    def _oversample_minority_class(self,df_dataset,oversampling_rate=-1,col_label='labels'):
        if oversampling_rate < 0:
            oversampling_rate = 10**10
        iPositiveCount = df_dataset.loc[:][col_label].sum()
        iNegativeCount = len(df_dataset)-iPositiveCount
        iClassCount = int(oversampling_rate * np.min([iPositiveCount,iNegativeCount]))
        iClassCount = int(np.min([iClassCount,np.max([iPositiveCount,iNegativeCount])]))
        oversampling_rate = iClassCount/np.min([iPositiveCount,iNegativeCount])
        
        condition_positive = df_dataset.loc[:][col_label]==1
        condition_negative = df_dataset.loc[:][col_label]==0
        df_positive = df_dataset[condition_positive]
        df_negative = df_dataset[condition_negative]
        
        n_repeat = iClassCount // len(df_positive)
        idx_positive = df_positive.loc[df_positive.index.repeat(n_repeat)].index
        n_sample = np.mod(iClassCount,len(df_positive))
        idx_positive = np.concatenate([idx_positive,df_positive[:n_sample].index])
        #idx_positive = np.concatenate([idx_positive,df_positive.sample(n_sample).index])
        
        n_repeat = iClassCount // len(df_negative)
        idx_negative = df_negative.loc[df_negative.index.repeat(n_repeat)].index
        n_sample = np.mod(iClassCount,len(df_negative))
        idx_negative = np.concatenate([idx_negative,df_negative[:n_sample].index])
        #idx_negative = np.concatenate([idx_negative,df_negative.sample(n_sample).index])
        
        indices = np.hstack([idx_positive,idx_negative])
        return [df_dataset.loc[indices].reset_index(drop=True),oversampling_rate]   #[df_dataset.loc[indices].sample(frac=1).reset_index(drop=True),oversampling_rate]

    def  _buildTestSet(self,df_annotated,threshold):

        df_dataset = self.df_dataset

        n_positive = len(df_dataset[df_dataset['weak_soft_label']>threshold])
        n_negative = len(df_dataset[df_dataset['weak_soft_label']<=threshold])
        negative_ratio = n_negative/n_positive
        
        df_positive = df_annotated[df_annotated['labels']==1]
        df_negative = df_annotated[df_annotated['labels']==0]

        if negative_ratio > 1:
            n = int(len(df_positive) * negative_ratio)
            df_negative_samples = df_negative.sample(n,replace=True)
            df_positive_samples = df_positive
        else:
            n = int(len(df_negative) / negative_ratio)
            df_positive_samples = df_negative.sample(n,replace=True)
            df_negative_samples = df_positive
        return pd.concat([df_positive_samples,df_negative_samples])

    def _train_test_split(self,df):

        max_test_size = self._getConfigParameter('test_size') #self.dConfig['test_size']
        test_size = 0
        positive_label_count = len(df[df['labels']==1])
        negative_label_count = len(df[df['labels']==0])
        minClassValue = np.min([positive_label_count,negative_label_count])
        if minClassValue > 1:
            min_test_size = 1/minClassValue
            test_size = np.max([min_test_size,max_test_size])
        else:
            return [df,None]
        df_train, df_test = train_test_split( df, test_size=test_size, random_state=42, stratify = df['labels'])
        return [df_train, df_test]

    def _get_weak_label_threshold(self):
        df_labeled = self._get_sub_set('labeled')
        true_labels = df_labeled['labels'].to_numpy()
        weak_soft_labels = df_labeled['weak_soft_label'].to_numpy()
        p_weak_soft_label = weak_soft_labels[true_labels==1]
        n_weak_soft_label = weak_soft_labels[true_labels==0]

        if (len(p_weak_soft_label) > 0) & (len(n_weak_soft_label) > 0):
            weak_label_threshold = np.mean([np.mean(p_weak_soft_label),np.mean(n_weak_soft_label)])
        else:
            weak_label_threshold = self._getConfigParameter('backup_threshold')

        return weak_label_threshold

    def _buildTrueAndWeakLabelDataset(self,weak_label_threshold,oversampling_rate):

        ##############################TRUE LABELS##############################
        df_labeled = self._get_sub_set('labeled')
        df_true_label_org = pd.DataFrame(df_labeled[['id','text','labels']]).sample(frac=1)
        df_train_true_label,df_test_true_label =self._train_test_split(df_true_label_org) 
        df_train_true_label = self._oversample(df_train_true_label,max_oversampling=oversampling_rate)

        df_true_label_org['is_test'] = True
        df_true_label_org.loc[df_train_true_label.index,['is_test']] = False
        df_test_true_label = df_true_label_org[df_true_label_org['is_test']]

##############################WEAK LABELS##############################
        
        #weak_label_threshold = self._get_weak_label_threshold()
        self.df_dataset.loc[:,['weak_label']] = (self.df_dataset['weak_soft_label'] > weak_label_threshold) * 1
        df_unlabeled = self._get_sub_set('unlabeled')
        df_weak_label = pd.DataFrame(df_unlabeled[['id','text','weak_label']].to_numpy(),columns=['id','text','labels']).sample(frac=1)
        df_weak_label = self._oversample(df_weak_label,max_oversampling=1)

        return df_train_true_label, df_test_true_label, df_weak_label

    def _train_classifier(self,df_train_true_label, df_test_true_label, df_weak_label,dClf):

        #reduce weak labels according to weak_label_ratio
        if dClf['weak_label_ratio'] < 1:
            _, df_weak_label_extract = train_test_split( df_weak_label, test_size=dClf['weak_label_ratio'], random_state=42, stratify = df_weak_label['labels'].to_numpy())
        else:
            df_weak_label_extract = df_weak_label

        self._printIt(f'Train Classifier {dClf["idx"]}: {len(df_weak_label_extract)} weak labels (weak_label_ratio: {dClf["weak_label_ratio"] }, weak_label_threshold:{dClf["weak_label_threshold"]}), {len(df_train_true_label)} true labels (oversampling_rate: {dClf["oversampling_rate"] })')
##############################MIXUP TRUE LABELS AND WEAK LABELS##############################
        df_train_mix = pd.concat([df_train_true_label,df_weak_label_extract])

        # import pdb
        # pdb.set_trace()

        dClf['true_weak_test_idxs'] = {'true': df_train_true_label.index, 'weak': df_weak_label_extract.index, 'test': df_test_true_label.index}

        last_history = dClf['clf'].train_loop(df_train_mix, df_test_true_label)
        dClf['f1_cont'] = last_history['f1_cont']
        dClf['last_history'] = last_history

    def _getNoiseFactorByF1Score(self,f1_score=0):
        
        
        # if f1Score > 0:
        #     import pdb
        #     pdb.set_trace()

        quantile_95 = self._getConfigParameter('quantile_95')
        scale = (quantile_95)/(1.96) #(quantile_95 - 0(mean)) / z

        annotationRatioFactor = self._getConfigParameter('annotationRatioFactor')
        exponentByAnnotationRatio = np.power(len(self.df_dataset)/len(self._get_sub_set('labeled')),annotationRatioFactor)

        scale *= (1-np.power(f1_score,exponentByAnnotationRatio)) #IDEA: high f1Score low noise, many anotations low noise
        #1 mean
        noise = np.random.normal(loc=0, scale=scale)
        if noise > quantile_95: #check upper bound: example 2.05 gets 1.95
            noise = quantile_95 - (noise - quantile_95) #
        if noise < -quantile_95: #check lower bound
            noise = -quantile_95 - (noise + quantile_95) 

        #example: noise is gausian distributed from -1 and 1 (with lower and upper bound)
        #0.5 => 1.5; -0.5 => 1/(1,5)
        noiseFactor = (1 + noise) if noise > 0 else (1 / (1+np.abs(noise)))
        #print(f'f1_score/annotationRatioFactor/exponentByAnnotationRatio/scale/noise/noiseFactor:{f1_score}/{annotationRatioFactor}/{exponentByAnnotationRatio}/{scale}/{noise}/{noiseFactor}')
        return noiseFactor
    
    def _updateProtocol(self,result):
        self.protocol.append({'annotations': str(len(self._get_sub_set('labeled'))), 
                              'f1_score': str(result['f1_cont']), 
                              'classifiers': [{'weak_label_threshold': str(clf['weak_label_threshold']),
                                               #'weak_label_ratio': str(clf['weak_label_ratio']),
                                               #'oversampling_rate': str(clf['oversampling_rate']),
                                               #'is_query_lead': clf['is_query_lead'],
                                               'f1_score': str(clf['f1_cont']) ,
                                               #'test_idxs': [int(x) for x in clf['true_weak_test_idxs']['test'].to_numpy()],
                                               #'true_idxs': [int(x) for x in clf['true_weak_test_idxs']['true'].to_numpy()], 
                                               #'weak_idxs': [int(x) for x in clf['true_weak_test_idxs']['weak'].to_numpy()]  
                                               } for clf in self.clfs] }) 

    def getProtocol(self):
        #change dtypes
        return self.protocol

    def saveProtocol(self):
        dProtocol = { 'data': self.protocol }
        protocolPath = Path(f'data/activeLearning/protocols/{ str(int(time.time()*10**7) ) }')
        protocolPath.parent.mkdir(parents=True, exist_ok=True)
        with open(protocolPath,'w') as outfile:
            json.dump(dProtocol, outfile)

    def eval(self,df_eval):
        clf = self._getClf('is_query_lead')['clf']
        result = clf.eval(df_eval)
        self._updateProtocol(result)

        return result
    def save(self):
        pass
    def load(self):
        pass