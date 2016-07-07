import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

DISTRICT_NAMES = ["Печерский р-н", "Соломенский р-н", "Деснянский р-н", "Шевченковский р-н", "Голосеевский р-н", "Дарницкий р-н", "Подольский р-н", "Святошинский р-н", "Оболонский р-н", "Днепровский р-н", "Киевский р-н"] 

DETAILS = ['floor_count', 'floor', 'rooms', 'without_fee', 'square', 'live_square', 'kitchen_square']

def district_vector(district_name):
    vec = [0]*len(DISTRICT_NAMES)
    try:
        vec[DISTRICT_NAMES.index(district_name)] = 1
    except: 
        pass
    return vec

def extract_data(flat):
    X = []
    for d in DETAILS:
        try:
            X += [flat['details'][d]['value']]
        except:
            X += [0]
    X += district_vector(flat['district'])
    y = flat['price']
    return [X, y]

#data
cleaned_data = [extract_data(i) for i in data]
X = [i[0] for i in cleaned_data]
y = [i[1] for i in cleaned_data]
edge = int(len(X)*0.8)
train_X = X[:edge]
train_y = y[:edge]
test_X = X[edge:]
test_y = y[edge:]

regr = linear_model.LinearRegression()
regr.fit(train_X, train_y)
print('Coefficients: \n', regr.coef_)

print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(test_X) - test_y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(test_X, test_y))

# Plot outputs
test_X_plot = [i for i in range(0, len(test_X))]
plt.scatter(test_X_plot, test_y,  color='black')
plt.plot(test_X_plot, regr.predict(test_X), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
