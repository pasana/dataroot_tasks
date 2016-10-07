import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn import cross_validation

DISTRICT_NAMES_RU = ["Печерский", "Соломенский", "Деснянский", "Шевченковский", "Голосеевский", "Дарницкий", "Подольский", "Святошинский", "Оболонский", "Днепровский"] 

DISTRICT_NAMES_UA = ["Печерський", "Солом'янський", "Деснянський", "Шевченківський", "Голосіївський", "Дарницький", "Подільський", "Святошинський", "Оболонський", "Дніпровський"] 

DETAILS = ["square", "kitchen_square","live_square", "rooms", "floor_count", "floor"]

USD = 24.5
EUR =  26.9

DIVISION_VALUE = 0.85

def district_vector(district_name):
    vec = [0]*len(DISTRICT_NAMES_RU)
    try:
        name = district_name.split()[0].encode('utf-8')
        vec[DISTRICT_NAMES_RU.index(name)] = 1
    except ValueError:
        try:
            vec[DISTRICT_NAMES_UA.index(name)] = 1
        except: 
            pass
    except:
        pass
    return vec

def extract_data(flat):
    X = []
    for d in DETAILS:
        try:
            X += [float(flat[d])]
        except:
            X += [0]
    if flat["without_fee"] == "False":
        X += [0]
    else:
        X += [1]
    try:
        X += district_vector(flat['district'])
    except:
        X += [0] * 10
    y = float(flat['price'])
    if flat['currency'].lower().encode() == 'usd':
            y *= USD
    if flat['currency'].lower().encode() == 'eur':
            y *= EUR
    return [X, y]

regr = linear_model.LinearRegression()

with open('places_kiev.json') as data_file:
    json_data = json.load(data_file)

#filter city Kiyv
data = [i for i in json_data if i['city'].encode('utf-8') in ["Киев","Київ"]]

#type 1
data_1 = [i for i in data if i['type'] == 1]
cleaned_data_1 = [extract_data(i) for i in data_1]
cleaned_data_1 = [i for i in cleaned_data_1 if not(i[0][0] == 0)]
X_1 = np.array([i[0] for i in cleaned_data_1])
#X_1 = preprocessing.normalize(X_1, norm='l2')
y_1 = np.array([i[1] for i in cleaned_data_1])
train_X_1, test_X_1, train_y_1, test_y_1 = cross_validation.train_test_split(X_1, y_1, test_size = 0.23, random_state = 5)
print "Total: %d, train: %d, test: %d" %(len(X_1), len(train_X_1), len(test_X_1))

regr.fit(train_X_1, train_y_1)
print('Coefficients: \n', regr.coef_)
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(test_X_1) - test_y_1) ** 2))
print('Variance score: %.4f' % regr.score(test_X_1, test_y_1))

res_1 = np.array([ [ regr.predict(test_X_1[i]), test_y_1[i],  regr.predict(test_X_1[i])/test_y_1[i]*100] for i in range(0, len(test_X_1))])
res_1 = res_1[np.argsort(res_1[:,2])]
for r in res_1:
    print "%d \t %d \t %d %%" % (r[0], r[1], r[2])

#type 2
data_2 = [i for i in data if i['type'] == 2]
cleaned_data_2 = [extract_data(i) for i in data_2]
cleaned_data_2 = [i for i in cleaned_data_2 if not(i[0][0] == 0)]
cleaned_data_2 = [i for i in cleaned_data_2 if 1 in i[0][-10:]]
X_2 = np.array([i[0] for i in cleaned_data_2])
#X_2 = preprocessing.normalize(X_2, norm='l1')
y_2 = np.array([i[1] for i in cleaned_data_2])
train_X_2, test_X_2, train_y_2, test_y_2 = sklearn.cross_validation.train_test_split(X_2, y_2, test_size = 0.18, random_state = 5)
print "Total: %d, train: %d, test: %d" %(len(X_2), len(train_X_2), len(test_X_2))

regr.fit(train_X_2, train_y_2)
print('Coefficients: \n', regr.coef_)
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(test_X_2) - test_y_2) ** 2))
print('Variance score: %.2f' % regr.score(test_X_2, test_y_2))

res_2 = np.array([ [ regr.predict(test_X_2[i]), test_y_2[i],  regr.predict(test_X_2[i])/test_y_2[i]*100] for i in range(0, len(test_X_2))])
res_2 = res_2[np.argsort(res_2[:,2])]
for r in res_2:
    print "%d \t %d \t %d %%" % (r[0], r[1], r[2])


# Plot outputs
test_X_plot = [i for i in range(0, len(test_X_1))]
plt.scatter(test_X_plot, test_y_1,  color='black')
plt.plot(test_X_plot, regr.predict(test_X_1), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


test_X_plot = [i for i in range(0, len(test_X_2))]
plt.scatter(test_X_plot, test_y_2,  color='black')
plt.plot(test_X_plot, regr.predict(test_X_2), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
