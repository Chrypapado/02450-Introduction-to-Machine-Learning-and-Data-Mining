#Import script
from import_data import *

#Select continuous attributes
cont = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

#Print the mean
print('The mean for each attribute:', '\n', data[cont].mean(), '\n')

#Print the standard deviation
print('The standard deviation for each attribute:', '\n', data[cont].std(), '\n')

#Print the variance
print('The variance for each attribute:', '\n', data[cont].var(), '\n')

#Print the minimum values
print('The minimum value for each attribute:', '\n', data[cont].min(), '\n')

#Print the maximum values
print('The maximum value for each attribute:', '\n', data[cont].max(), '\n')