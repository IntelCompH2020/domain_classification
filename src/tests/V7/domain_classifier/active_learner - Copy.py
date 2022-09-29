# try:
from .custom_model import CustomModel
from .query_strategy import BalancedRandomSampling
import numpy as np
from copy import copy, deepcopy


class ActiveLearner(object):
    def __init__(self, clf, queryStrategy, df_dataset=[], dConfig = None):
        self.clf = clf
        # import pdb
        # pdb.set_trace()
        #self.clf_org = deepcopy(clf)
        self.queryStrategy = queryStrategy
        self.dConfig = dConfig if dConfig != None else self._getDefaultConfig()

        if len(df_dataset) > 0:
            self.df_dataset = df_dataset
        else:
            self.df_dataset = self.load()

        #add Column
        self.df_dataset.loc[:,['labels']] = -1
        self.df_dataset.loc[:,['was_used_for_training']] = False
        self.indices_queried = []
    def _getDefaultConfig(self):
        return { 'path': 'data/activeLearning/',
                 'max_oversampling': 10 }
    def query(self, num_samples=10):
        # import pdb
        # pdb.set_trace()
        condition_unlabeld = (self.df_dataset['labels'] == -1) #& (self.df_dataset['was_used_for_training'] == False)
        self.indices_queried = self.queryStrategy.query(deepcopy(self.clf), self.df_dataset[condition_unlabeld],num_samples)

        # import pdb
        # pdb.set_trace()
        return self.indices_queried
    def oversample(self):
        max_oversampling = self.dConfig['max_oversampling']
        df_oversample = self.df_dataset.copy()
        #condition_oversample = (df_oversample['was_used_for_training'] == False) & (df_oversample['labels'] != -1)
        condition_oversample = (df_oversample['labels'] != -1)
        iPositiveCount = self.df_dataset[condition_oversample]['labels'].sum()
        iNegativeCount = len(self.df_dataset[condition_oversample])-iPositiveCount

        iClassCount = np.min([np.max([iPositiveCount,iNegativeCount]),iPositiveCount*max_oversampling,iNegativeCount*max_oversampling])

        condition_positive =  (df_oversample['labels'] == 1) & (df_oversample['was_used_for_training'] == False)
        condition_negative =  (df_oversample['labels'] == 0) & (df_oversample['was_used_for_training'] == False)
        idx_positive = df_oversample[condition_positive].sample(iClassCount, replace = True).index 
        idx_negative = df_oversample[condition_negative].sample(iClassCount, replace = True).index

        indices = np.hstack([idx_positive,idx_negative])
        return df_oversample.loc[indices]
    def update(self, y, bSave=True):
        # import pdb
        # pdb.set_trace()
        self.df_dataset.loc[self.indices_queried,['labels']] = y
        # import pdb
        # pdb.set_trace()
        df_train_subset = self.oversample()

        #test_size = 0.2
        #train_test_split

        # print(len(df_train_subset))
        if len(df_train_subset) > 0:

            self.clf.train_loop(df_train_subset)
            self.df_dataset.loc[df_train_subset.index,['was_used_for_training']] = True
            if bSave:
                self.save()
            return True 
        return False
    

    def eval(self,df_eval):
        #result = self.clf.eval(df_eval)
        #condition_labeld = (self.df_dataset['labels'] != -1)
        # import pdb
        # pdb.set_trace()
        #result = {**{ 'labeld': str(len(self.df_dataset.loc[condition_labeld])) }, **result }
        return self.clf.eval(df_eval)
    def save(self):
        pass
    def load(self):
        pass



        # indices_train = self._determineTrainIndices()
        # if len(indices_train) > 0
            




        #return np.random.choice(indices_unlabeled, size=n, replace=False)