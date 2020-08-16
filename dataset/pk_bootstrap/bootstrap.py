import os, sys, time, itertools
import pandas as pd
import sklearn.linear_model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def main():
    data_core = pd.read_csv('pk.labeled.tsv',sep='\t',header=None)
    data_extra = pd.read_csv('pk.nlm180.tsv',sep='\t',header=None)

    X_core = data_core[1].tolist()
    y_core = data_core[0].tolist()

    X_extra = data_extra[4].tolist()
    y_extra = [ y if isinstance(y,str) else 'NULL' for y in data_extra[3].tolist() ]

    y_extra_yep = data_extra[2].tolist()

    vx = CountVectorizer(ngram_range=(1,4),
                        token_pattern=r'\b\w+\b', 
                        min_df=1)
    vx.fit(X_core + X_extra)

    m1 = sklearn.linear_model.LogisticRegression(C=1.0)
    label_space = list(set(y_core))

    print('Starting with {} unlabeled examples'.format(len(y_extra)))

    X_aux, y_aux = aux_dataset(X_extra, y_extra)
    m1.fit(vx.transform(X_core + X_aux), y_core + y_aux)

    y_proba = m1.predict_proba(vx.transform(X_extra))
    y_pred = m1.predict(vx.transform(X_extra))
    
    confidence = np.max(y_proba,axis=-1)
    easy = np.where(confidence > 0.6)[0].tolist()

    easy = [ idx for idx in easy if y_extra[idx] == 'NULL' ]

    increase_codes = ['C54355', 'C54602', 'C54603', 'C54604', 'C54605',
                        'C54357','C54610','C54611','C54612','C54613']
    
    adding = 0
    for idx in easy:
        if y_pred[idx] in increase_codes and y_extra_yep[idx] == 'Increase_Interaction':
            y_extra[idx] = y_pred[idx]
            adding += 1
        elif y_pred[idx] not in increase_codes and y_extra_yep[idx] == 'Decrease_Interaction':
            y_extra[idx] = y_pred[idx]
            adding += 1
        else:
            y_extra[idx] = 'TODO'
    
    if len(y_aux) > 3:
        remains = list(y_extra).count('NULL')
        m1.fit(vx.transform(X_aux),y_aux)
        fit = m1.score(vx.transform(X_core),y_core)
        print('{} easy, +{} added, {} remaining :: fit> {:.2%}'
                .format(len(easy), adding, remains,fit))
    else:
        print('initialized')
    
    data_extra[3] = y_extra
    data_extra.to_csv('pk.nlm180.tsv',sep='\t',index=False,header=False)
    
def aux_dataset(X,y, save_idx = None):
    X_aux = []
    y_aux = []
    new_idx = None
    for i in range(len(y)):
        if y[i] != 'NULL':
            X_aux.append(X[i])
            y_aux.append(y[i])
            if save_idx is not None and i == save_idx:
                new_idx = len(y_aux)-1
    
    if new_idx is not None:
        return X_aux, y_aux, new_idx
    else:
        return X_aux, y_aux

def brute_search(vx, X_core, y_core, X_extra, y_extra, exid, label_space):
    m = sklearn.linear_model.LogisticRegression(C=10)
    scores = []
    for c in label_space:
        y_extra_tmp = [ y if i != exid else c for i, y in enumerate(y_extra) ]
        X_aux, y_aux, new_id = aux_dataset(X_extra, y_extra_tmp, save_idx = exid)
        sample_weight = np.ones(len(y_aux))
        sample_weight[new_id] = len(y_aux)/10.0
        m.fit(vx.transform(X_aux),y_aux, sample_weight=sample_weight)
        scores.append(m.score(vx.transform(X_core),y_core))
    
    maximum = np.max(scores)
    if len(np.where(scores >= maximum)[0]) > 1:
        return None
    else:
        return label_space[np.argmax(scores)]

if __name__ == '__main__':
    main()