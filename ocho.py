import scipy
import numpy as np
import csv
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import statsmodels
import pandas
from datetime import datetime
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class Dato:
    def __init__(self, date, location, product, sa_quantity, temp_mean, temp_max, temp_min, sunshine_quant, price):
        self.date = date
        self.location = location
        self.product = product
        self.sa_quantity = sa_quantity
        self.temp_mean = temp_mean
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.sunshine_quant = sunshine_quant
        self.price = price

dateparse = lambda x: pandas.datetime.strptime(x, '%m/%d/%Y')
series = pandas.read_csv('input_data_train.csv', header=0, index_col = False, parse_dates=['date'], date_parser=dateparse)

series = series[0:50000]

#print(uniqDate)

#series = series.sort_values(['date', 'location'], axis=1, ascending=True, inplace=True)
#series2 = series.sort_values(by=['date', 'location'])
#ax y series2 = series2.reset_index(drop=True)

#seriesLocation = series.sort_values(by=['location'])
#seriesLocation = seriesLocation.reset_index(drop=True)

#seriesLocationUniq = series['location'].unique()
'''
for i in range(len(seriesLocationUniq)):

    print(series2['location'][i])
    
    currLoc = series2['location'][i]

    try:
        nextLoc = series2['location'][i+1]

        if(currLoc == nextLoc):

            print(series2['date'][10])
'''
ventasTotales = 0
csvData = [['date', 'location', 'sales', 'temp_mean', 'temp_max', 'temp_min', 'sunshine_quant', 'price']]

for i in range(len(series2)):
    currDate = series2['date'][i]
    currLoc = series2['location'][i]

    try:
        nextDate = series2['date'][i+1]
        nextLoc = series2['location'][i+1]
        if(currDate == nextDate ):
            if(currLoc == nextLoc):
                ventasTotales = ventasTotales + series2['sa_quantity'][i]
            else:
                ventasTotales = ventasTotales + series2['sa_quantity'][i]
                print(str(currDate) + ' ' + str(currLoc) + ' ' + str(ventasTotales))
                csvData.append([currDate, currLoc, ventasTotales, series2['temp_mean'][i], series2['temp_max'][i], series2['temp_min'][i], series2['sunshine_quant'][i], series2['price'][i]])
                ventasTotales = 0
        else:
            ventasTotales = ventasTotales + series2['sa_quantity'][i]
            print(str(currDate) + ' ' + str(currLoc) + ' ' + str(ventasTotales))
            csvData.append([currDate, currLoc, ventasTotales, series2['temp_mean'][i], series2['temp_max'][i], series2['temp_min'][i], series2['sunshine_quant'][i], series2['price'][i]])
            ventasTotales = 0
    except:
        break


print(csvData)
a = np.asarray(csvData)

with open("cvsExport.csv","w", newline='') as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(a)

    
'''
        else:

    except:
        break
    
'''

'''
#datos = []
print(series2['date'][10])

ventasTotales = 0

for i in range(len(series2)):
    currDate = series2['date'][i]
    try:
        nextDate = series2['date'][i+1]
        if(currDate == nextDate):
            ventasTotales = ventasTotales + series2['sa_quantity'][i]
        else:
            ventasTotales = ventasTotales + series2['sa_quantity'][i]
            print(str(currDate) + ' ' + str(ventasTotales))
            ventasTotales = 0
    except:
        break
    #if(currDate != series['date'][i+1]):
        #print(str(currDate) + ' ' + str(ventasTotales))
        #ventasTotales = 0
    #print(ventasTotales)
 '''   

'''
#for i in range(len(series)):
    #datos.append(Dato(series['date'][i], series['location'][i], series['product'][i], series['sa_quantity'][i], series['temp_mean'][i], series['temp_max'][i], series['temp_min'], series['sunshine_quant'][i], series['price'][i]))
    #print(i)

#datos = datos.sort(key=lambda x: x[0])

#print(series)
'''


'''
for i in range(len(uniqDate)):
    ventasDelDia = 0
    for j in range(len(series)):
        if series['date'][j] == uniqDate[i]:
            ventasDelDia = ventasDelDia + series['sa_quantity'][j]
            #print(str(series['location'][j]) + ' ' + str(series['product'][j]) + ' ' + str(uniqDate[i]) + ' ' + str(series['sa_quantity'][j])) 
    print(str(uniqDate[i]) + ' ' + str(ventasDelDia))

'''


#################################################################################
'''
for i in range((len(uniqDate))):
    uniqDate = series['date'].unique()
    dateActual = uniqDate[i]
    print(dateActual)

    series.loc[series['date'].isin(dateActual[i])]

'''

#################################################################################


'''
for i in range(len(locations)):

    series = pandas.read_csv(locations[i-1] + '.csv', header=0, parse_dates=['date'], date_parser=dateparse)
    series.drop(['location'], axis=1, inplace=True)
    print(series)

    for i in range(len(series.product.unique())):

        

    series.columns = ['date', 'product','sa_quantity','temp_mean','temp_max','temp_min','sunshine_quant','price']
    series.index.name = 'date'
    series.fillna(-99999, inplace=True)

    # Assign categories
    #series['location'] = series['location'].astype('category').cat.codes
    series['product'] = series['product'].astype('category').cat.codes

    print(series)

    series = series[0:150]

    #std_dev = 2000

    #series = series[(np.abs(stats.zscore(series)) < float(std_dev)).all(axis=1)]

    # Assign x and y values to get sales
    X = np.asarray(series[['temp_mean', 'temp_max', 'temp_min', 'sunshine_quant', 'price']])
    Y = np.asarray(series['sa_quantity'])

    scores = []
    coefs = []

    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle= True)
        lineReg = linear_model.Ridge (alpha = .5)
        lineReg.fit(X_train, y_train)
        scores.append(lineReg.score(X_test, y_test))
        coefs.append(lineReg.coef_)
    print('\nRidge Regression')
    print(np.mean(scores))
    print(np.mean(coefs, axis=0))


    plt.plot(lineReg.predict(X_test))
    plt.plot(y_test)
    plt.show()


# Load Data
dateparse = lambda x: pandas.datetime.strptime(x, '%m/%d/%Y')
series = pandas.read_csv('input_data_train.csv', header=0, parse_dates=['date'], index_col=[2], date_parser=dateparse)
series.drop(['event'], axis=1, inplace=True)
print(series)

series.columns = ['location', 'product', 'sa_quantity','temp_mean','temp_max','temp_min','sunshine_quant','price']
series.index.name = 'date'
series.fillna(-99999, inplace=True)

print(series.head(5))

series.to_csv('training.csv')

# Assign categories
series['location'] = series['location'].astype('category').cat.codes
series['product'] = series['product'].astype('category').cat.codes

print(series)

series = series[0:1000]

#std_dev = 2000

#series = series[(np.abs(stats.zscore(series)) < float(std_dev)).all(axis=1)]

# Assign x and y values to get sales
X = np.asarray(series[['location', 'product', 'temp_mean', 'sunshine_quant', 'price']])
Y = np.asarray(series['sa_quantity'])

scores = []
coefs = []

for i in range(10000):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle= True)
    lineReg = linear_model.Ridge (alpha = .5, tol=10)
    lineReg.fit(X_train, y_train)
    scores.append(lineReg.score(X_test, y_test))
    coefs.append(lineReg.coef_)
print('\nRidge Regression')
print(np.mean(scores))
print(np.mean(coefs, axis=0))


plt.plot(lineReg.predict(X_test))
plt.plot(y_test)
plt.show()

scores = []
coefs = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle= True)
    lineReg = linear_model.Ridge (alpha = .5)
    lineReg.fit(X_train, y_train)
    scores.append(lineReg.score(X_test, y_test))
    coefs.append(lineReg.coef_)
print('\nRidge Regression')
print(np.mean(scores))
'''

'''
# Prepare Data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
'''
'''
plt.matshow(series.corr())
plt.xticks(np.arange(8), series.columns, rotation=90)
plt.yticks(np.arange(8), series.columns, rotation=0)
plt.colorbar()
plt.show()


# Walk-forward Validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):

	# Predict
	yhat = history[-1]
	predictions.append(yhat)
	# Observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# Report Performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
'''