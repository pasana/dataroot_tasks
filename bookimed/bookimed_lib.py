
# coding: utf-8

# In[1]:

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn import cross_validation

#new_ones
gd = [-0.14858283,  0.82626225,  0.72419524,  0.38033358, -0.41910666, -0.0017822, 8.504 ]
gd_1 = [ 0.38720291,  0.6170334 ,  1.3575343 ,  0.06852735,  0.2979859 , 0.01196117, 7.98339173738]
gd_2 = [-0.34151137, -0.43818642,  1.48721481, -0.65192724,  0.55046177, 0.01478482, 8.28216808672]
gd_3 = [ 0.22044356, -0.03968407, -0.2108969 ,  0.25643846,  0.17997737, 0.00104368, 7.83839721743]
gd_4 = [  5.08011543e-01,   9.61926105e-01,   1.18588486e+00, -1.22263084e-01,  -9.58909993e-01,  -7.53080350e-04, 8.31196433005]
gd_5 = [ -3.31640563e-01,   7.05312018e-02,   7.66855734e-01, 4.78101072e-16,   1.30142649e-01,   1.54756015e-03, 8.46744569939]

ed = [-0.2529,-0.5355,-0.2411,0.1319,8.84130721381]
ed_1 = [1.2004,-2.1618,0.0,1.6412,8.67028085972]
ed_2 = [-1.1872,1.3869,0.0,-0.5365,8.70100044872]
ed_3 = [-1.368,0.5593,0.0,0.4327,7.9939688955]
ed_4 = [-2.5859,1.1579,-0.2749,0.0117,9.88449832052]
ed_5 = [-1.4355,0.2174,0.0,0.0907,8.85713476559]

gp = [ -5.79394428, -16.03602744,  16.81469781,  14.56978414]
gp_1 = [ -8.40030539, -11.98444574,  18.38346948,  10.71352994]
gp_2 = [-10.17055835,  -2.9336299 ,  22.35123782,   0.54050688]
gp_3 = [ -0.25488576, -21.560419  ,  13.17436164,  17.98284953]
gp_4 = [ -5.00790327, -13.37337951,  15.23204126,  13.27778045]
gp_5 = [ -8.26148188, -12.47220116,  20.91674746,   7.82969123]


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


# In[4]:

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


# In[ ]:

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


# In[5]:

def process_with(X,y, info=False, short=False, return_short = False, return_long = False, new_coef = [], ts=0.2):
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
    if return_long:
        return {"absolute_train": np.mean(abs(regr.predict(train_X) - train_y)),
               "absolute_test": np.mean(abs(regr.predict(test_X) - test_y)),
               "variance_train": regr.score(train_X, train_y),
               "variance_test": regr.score(test_X, test_y),
               ""
               "ts": ts}
    #for i in regr.coef_:
        #print "%.3f" % i
    #print "%.3f" % regr.intercept_
    return regr

# In[6]:

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


# In[7]:

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


# In[8]:

def get_X_sets(extract_data):
    X_all, y_all = [], []

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


# In[12]:

def get_clinic_names(t_estims_data, t_cancer_data):
    ids = [i['id'] for i in t_estims_data]
    if '0' in ids:
        ids.pop('0')
    clinic_names = []
    for id_ in ids:
        clinic_names += [ i["name_ru"] for i in t_cancer_data if i['id'] == id_ and len(i['doctors'])!= 0]
    return clinic_names

# In[14]:

def get_clinic_names_all():
    X_all, y_all = [], []

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[2]['clinics']
    with open('%s/меланома_все.json'%this_path) as data_file: #2
        cancer_data = json.load(data_file)[0]['clinics']
    clinic_names_1 = get_clinic_names(estims_data, cancer_data)

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[0]['clinics']
    with open('%s/рак_груди_все.json'%this_path) as data_file: #0
        cancer_data = json.load(data_file)[0]['clinics']
    clinic_names_2 = get_clinic_names(estims_data, cancer_data)

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[4]['clinics']    
    with open('%s/рак_простаты_все.json'%this_path) as data_file: #4
        cancer_data = json.load(data_file)[0]['clinics']
    clinic_ids = [i['id'] for i in estims_data]
    estims_data.pop(clinic_ids.index('0'))
    clinic_names_3 = get_clinic_names(estims_data, cancer_data)

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[1]['clinics']     
    with open('%s/рак_шейки_матки_все.json'%this_path) as data_file: #1
        cancer_data = json.load(data_file)[0]['clinics']
    clinic_names_4 = get_clinic_names(estims_data, cancer_data)

    with open('./max/estims.json') as data_file: 
        estims_data = json.load(data_file)[3]['clinics']         
    with open('%s/рак_щитовидки_все.json'%this_path) as data_file: #3
        cancer_data = json.load(data_file)[0]['clinics']
    clinic_names_5 = get_clinic_names(estims_data, cancer_data)

    
    return clinic_names_1 + clinic_names_2 + clinic_names_3 + clinic_names_4 + clinic_names_5


# In[ ]:



