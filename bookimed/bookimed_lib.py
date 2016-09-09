
# coding: utf-8

# In[ ]:

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn import cross_validation


# In[2]:

path_1 = './max/'
path_2 = './max20160907'
this_path = path_2


# In[3]:

def get_X_y_from(t_extract_data, t_data, t_estims_data):
    clinic_ids = [i['id'] for i in t_data]
    t_cleaned_data = [t_extract_data(t_data[clinic_ids.index(i['id'])], i) for i in t_estims_data]
    X = sum([i[0] for i in t_cleaned_data],[])
    y = sum([i[1] for i in t_cleaned_data],[])
    return X,y


# In[ ]:

def get_X_from(t_data, t_extract_data):
    t_cleaned_data = []
    clinic_names = []
    ids = []
    for i in t_data:
        if i['doctors']!=[]:
            t_cleaned_data += [t_extract_data(i,[])]
            clinic_names += [unicode(i['name_ru'])] * len(i['doctors'])
            ids += [int(i['id'])] * len(i['doctors'])
    X = sum([i for i in t_cleaned_data],[])
    return X, clinic_names, ids

def get_best_ts(X, y):
    results = pd.DataFrame(columns = ['variance_train', 'variance_test', 'absolute_train', 'absolute_test', 'ts'])
    for i in range(1,7):
        regr = process_with(X, y, return_long=True, ts=i/10.0)
        if ((regr['variance_train'] == 1) and (round(regr['absolute_train'],2) == 0)): continue
        results = results.append(regr,ignore_index=True)
        results['score'] = results.apply(lambda row: abs(((row['absolute_train'] + row['absolute_test'])/2.0)/
                                 ((1 - row['variance_train'])/2.0)*
                                 ((1 - row['variance_test']/2.0)))*
                                 abs((row['absolute_train'] - row['absolute_test'])), axis=1)

    if len(results): return results['ts'][results['score'].argmin()]
    else: return 0

# In[7]:

def process_with(X,y, info=False, short=False, return_short = False, new_coef = [], ts=0.2):
    train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, test_size = ts, random_state = 3)
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    if new_coef != []:
        print "new coefs"
        regr.coef_ = new_coef
    regr.fit(train_X, train_y)
    if info:
        print "Total: %d, train: %d, test: %d" %(len(X), len(train_X), len(test_X))
        print("Residual sum of squares: %.2f"% np.mean((regr.predict(test_X) - test_y) ** 2))
        print("Train absolute: %.2f"% np.mean(abs(regr.predict(train_X) - train_y)))
        print("Test absolute: %.2f"% np.mean(abs(regr.predict(test_X) - test_y)))
        print("Absolute to mean: %.2f%%"% (np.mean(abs(regr.predict(test_X) - test_y))/np.mean(test_y)*100))
        print('Train variance score: %.2f' % regr.score(train_X, train_y))
        print('Test variance score: %.2f' % regr.score(test_X, test_y))
    if short:
        print "Total: %d, train: %d, test: %d" %(len(X), len(train_X), len(test_X))
        print "%.3f" % np.mean(abs(regr.predict(train_X) - train_y))
        print "%.3f" % np.mean(abs(regr.predict(test_X) - test_y))
        print "%.3f" % (np.mean(abs(regr.predict(test_X) - test_y))/np.mean(test_y)*100)
        print "%.3f" % regr.score(train_X, train_y)
        print "%.3f" % regr.score(test_X, test_y)
    if return_short:
        return np.mean(abs(regr.predict(test_X) - test_y)),regr.score(test_X, test_y)
    for i in regr.coef_:
        print "%.3f" % i
    #print "%.3f" % regr.intercept_
    return regr


# In[ ]:

def pack(X, gd, ed, gp):
    new_X = []
    for x in X:
        new_X+= [sum([
            [round(sum(np.array(gd)*np.array(x[0:7])),5)],
            [round(sum(np.array(ed)*np.array(x[7:12])),5)],
            [round(sum(np.array(gp)*np.array(x[12:16])),5)],
            x[16:]
        ],[])]
    return new_X


# In[ ]:

def seq_procent(input_array):
    new_pos=np.array(input_array)
    straight = np.array([i for i in range(max(new_pos), min(new_pos)-1, -1)])
    inversions = abs(straight - new_pos)
    while not np.array_equal(inversions, np.zeros(len(inversions))):
        new_pos=np.delete(new_pos, inversions.argmax())
        new_pos = [10 - sorted(new_pos, reverse=True).index(x) for x in new_pos]
        straight = np.array([i for i in range(max(new_pos), min(new_pos)-1, -1)])
        inversions = abs(straight - new_pos)
    return 100*len(new_pos)/float(len(input_array))


# In[6]:

def get_X_sets(extract_data):
    X_all, y_all = [], []

    this_path = path_2
    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[2]['clinics']
    with open('%s/меланома_все.json'%this_path) as data_file: #2
        cancer_data = json.load(data_file)[0]['clinics']
    X_1, y_1 = get_X_y_from(extract_data, cancer_data, estims_data)
    X_all+=X_1
    y_all+=y_1

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[0]['clinics']
    with open('%s/рак_груди_все.json'%this_path) as data_file: #0
        cancer_data = json.load(data_file)[0]['clinics']
    X_2, y_2 = get_X_y_from(extract_data, cancer_data, estims_data)
    X_all+=X_2
    y_all+=y_2

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[4]['clinics']    
    with open('%s/рак_простаты_все.json'%this_path) as data_file: #4
        cancer_data = json.load(data_file)[0]['clinics']
    clinic_ids = [i['id'] for i in estims_data]
    estims_data.pop(clinic_ids.index('0'))
    X_3, y_3 = get_X_y_from(extract_data, cancer_data, estims_data)
    X_all+=X_3
    y_all+=y_3

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[1]['clinics']     
    with open('%s/рак_шейки_матки_все.json'%this_path) as data_file: #1
        cancer_data = json.load(data_file)[0]['clinics']
    X_4, y_4 = get_X_y_from(extract_data, cancer_data, estims_data)
    X_all+=X_4
    y_all+=y_4

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[3]['clinics']         
    with open('%s/рак_щитовидки_все.json'%this_path) as data_file: #3
        cancer_data = json.load(data_file)[0]['clinics']
    X_5, y_5 = get_X_y_from(extract_data, cancer_data, estims_data)
    X_all+=X_5
    y_all+=y_5
    
    return X_all, X_1, X_2, X_3, X_4, X_5, y_all, y_1, y_2, y_3, y_4, y_5

