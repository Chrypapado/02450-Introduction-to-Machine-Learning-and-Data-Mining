#Libraries
import numpy as np

#Import script
from import_data import *

#Check how many sample have missing values
count_mv = sum([True for idx,row in data.iterrows() if any(row.isnull())])
print('There are {} samples with missing values'.format(count_mv), '\n')

#Print the missing values for every attribute
print(data.isnull().sum(),'\n')

#Remove some samples with missing values
data.drop(data.loc[(data['education'].isnull() == True)].index, inplace=True) 
data.drop(data.loc[(data['BPMeds'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['heartRate'].isnull() == True)].index, inplace=True)

#Remove samples with over two missing attributes
data.drop(data.loc[(data['BPMeds'].isnull() == True) & (data['totChol'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['BPMeds'].isnull() == True) & (data['glucose'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['totChol'].isnull() == True) & (data['BMI'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['totChol'].isnull() == True) & (data['glucose'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['BMI'].isnull() == True) & (data['glucose'].isnull() == True)].index, inplace=True)

#Number of samples dropped
modified_mv = sum([True for idx,row in data.iterrows() if any(row.isnull())])
deleted_mv = count_mv - modified_mv
print("{} samples with missing atrributes were deleted".format(deleted_mv), '\n')

#Fill the missing values with its column's mean value
cigsPerDay_mean = round(np.mean(data.loc[data['currentSmoker'] == 1]['cigsPerDay'])) #took the mean of the current smokers
data['cigsPerDay'].fillna(cigsPerDay_mean, inplace=True)

totChol_mean = np.mean(data['totChol'])
data['totChol'].fillna(totChol_mean, inplace=True)

BMI_mean = np.mean(data['BMI'])
data['BMI'].fillna(BMI_mean, inplace=True)

glucose_mean = np.mean(data['glucose'])
data['glucose'].fillna(glucose_mean, inplace=True)

#Number of samples modified
print("{} samples with missing atrributes were modified".format(modified_mv), '\n')

#Check for missing values
print('Are there any missing values? {}'.format(data.isnull().values.any()), '\n')