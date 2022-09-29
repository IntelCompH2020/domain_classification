from abc import ABC, abstractmethod
import numpy as np
import time

class QueryStrategy(ABC):
    @abstractmethod
    def query(self, clf, indices_unlabeled, n):
        pass

    @staticmethod
    def _validate_n(indices_unlabeled, n):
        return np.min([n,len(indices_unlabeled)])


class BalancedRandomSampling(QueryStrategy):
    def query(self, clf, _dataset, n, maxTime=10): #def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        n = self._validate_n(_dataset, n)

        maxSamples = 1000

        indices = _dataset.index.copy()
        positive_indices = []
        negative_indices = []
        class_count = n//2

        start = time.time()
        cResults = []
        cIndices = []
        while True:
            rand_indices = np.random.choice(indices, size=n, replace=False)
            cIndices = np.append(cIndices,rand_indices)

            results = clf.predict_proba(_dataset.loc[rand_indices])
            cResults = np.append(cResults,results)


            isSufficientPossitive = len(positive_indices) >= class_count
            isSufficientNegative = len(negative_indices) >= class_count

            if not isSufficientPossitive:
                positive_indices = np.append(positive_indices, rand_indices[results >= 0.5])
            if not isSufficientNegative:
                negative_indices = np.append(negative_indices, rand_indices[results < 0.5])

            if (isSufficientPossitive & isSufficientNegative) or (time.time()-start > maxTime):
                break

            mask = np.in1d(indices,rand_indices)
            rand_idx = (mask * np.arange(0,len(indices)))[mask]
            indices = np.delete(indices, rand_idx, axis=0)

        if not isSufficientPossitive:
            positive_indices =  cIndices[np.argsort(-1*cResults)]  
            print(cResults[np.argsort(cResults)])  
        if not isSufficientNegative:
            negative_indices =  cIndices[np.argsort(cResults)] 
            print(cResults[np.argsort(cResults)])    

        indices = (np.hstack([positive_indices[:class_count],negative_indices[:class_count]]))
        assert len(indices) == n
        return indices[np.random.choice(len(indices),len(indices),replace=False)]


    def __str__(self):
        return 'BalancedRandomSampling()'

class WeakSoftLabelTrustSampling(QueryStrategy):
    def query(self, clf, df_dataset, n, weak_soft_label_trust = 1): #def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):

        negative_count = n//2 #n_classes[0]
        positive_count = n//2 #n_classes[1]
        positive_indices,negative_indices = [],[]
        predict_proba_positive, predict_proba_negative = [], []

        #sample again if accidently the same index is drawn multiple times


        if clf == None:
        #weakSoftLabelSampling
            # print(f'Query top {positive_count} and bottom {positive_count} documents by weak soft label')
            predict_proba = df_dataset.loc[:,['weak_soft_label']].to_numpy().flatten()
            sorted_indices = df_dataset.index
            positive_indices = sorted_indices[:positive_count]
            negative_indices = sorted_indices[-negative_count:]
            predict_proba_positive = predict_proba[:positive_count]
            predict_proba_negative = predict_proba[-negative_count:]
            query_indices = (np.hstack([negative_indices,positive_indices]))
            predict_proba = (np.hstack([predict_proba_positive,predict_proba_negative]))
        else:
            predict_proba = clf.predict_proba(df_dataset)
            indices = np.argsort(np.abs(predict_proba - 0.5))[:n]
            query_indices = df_dataset.index[indices]
            
            # positive_sample_idxs = []
            # while len(positive_sample_idxs) < sample_amount:
            #     sample_idxs = np.random.exponential(scale=len(df_dataset)/(2*10), size=(sample_amount-len(positive_sample_idxs))).astype(int)
            #     sample_idxs = np.unique(sample_idxs)
            #     positive_sample_idxs = np.append(positive_sample_idxs,sample_idxs)
            # positive_sample_idxs = positive_sample_idxs[:sample_amount]
            # negative_sample_idxs = []
            # while len(negative_sample_idxs) < sample_amount:
            #     sample_idxs = np.random.exponential(scale=len(df_dataset)/(2*10), size=(sample_amount-len(negative_sample_idxs))).astype(int)
            #     sample_idxs = np.unique(sample_idxs)
            #     negative_sample_idxs = np.append(negative_sample_idxs,sample_idxs)
            # negative_sample_idxs = negative_sample_idxs[:sample_amount]


            # # print(f'Query top {positive_count} and bottom {positive_count} documents by result of predictions of the classifier which has the query lead')
            # if negative_count > 0:
            #     predict_proba_negative = clf.predict_proba(df_dataset[-1000:])
            #     negative_indices = df_dataset[-1000:].index[(predict_proba_negative).argsort()[:negative_count]]
            #     predict_proba_negative = predict_proba_negative[(predict_proba_negative).argsort()[:negative_count]]
            # if positive_count > 0:
            #     predict_proba_positive = clf.predict_proba(df_dataset[:1000])
            #     positive_indices = df_dataset[:1000].index[(-predict_proba_positive).argsort()[:positive_count]]
            #     predict_proba_positive = predict_proba_positive[(-predict_proba_positive).argsort()[:positive_count]]
            
        

        return [query_indices, predict_proba]


    def __str__(self):
        return 'WeakSoftLabelTrustSampling()'

class BalancedExtremSampling(QueryStrategy):
    def query(self, clf, df_dataset, n, retrained=True): #def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):

        #weak label ratio 

        n = self._validate_n(df_dataset, n)
        #print(f'retrained: { retrained }')
            # import pdb
            # pdb.set_trace()

        if 1 == 2: #clf != None:

            df_dataset = df_dataset.sample(frac=1).iloc[:6000]
            df_dataset.loc[:,['predict_proba']] = 0.

            import pdb
            pdb.set_trace()

            if retrained:
                df_dataset.loc[:,['predict_proba']] = clf.predict_proba(df_dataset)
                sorted_indices = df_dataset.iloc[np.argsort(-1*df_dataset['predict_proba'].to_numpy())].index
        else:
            df_dataset.loc[:,['predict_proba']] = df_dataset.loc[:,['weak_soft_label']].to_numpy()
            sorted_indices = df_dataset.index

       
        class_count = n//2

        

        indices = (np.hstack([sorted_indices[:class_count],sorted_indices[-class_count:]]))

        # import pdb
        # pdb.set_trace()
        return [indices, df_dataset.loc[indices,['predict_proba']] ]
        #return [indices[np.random.choice(len(indices),len(indices),replace=False)], df_dataset[['id','predict_proba']] ]


    def __str__(self):
        return 'BalancedExtremSampling()'