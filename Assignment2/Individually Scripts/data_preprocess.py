#Libraries
import pandas as pd
import numpy as np

#Import the Data
data = pd.read_csv("framingham.csv")

#Missing Data
data.drop(data.loc[(data['education'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['BPMeds'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['heartRate'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['BPMeds'].isnull() == True) & (data['totChol'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['BPMeds'].isnull() == True) & (data['glucose'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['totChol'].isnull() == True) & (data['BMI'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['totChol'].isnull() == True) & (data['glucose'].isnull() == True)].index, inplace=True)
data.drop(data.loc[(data['BMI'].isnull() == True) & (data['glucose'].isnull() == True)].index, inplace=True)

cigsPerDay_mean = round(np.mean(data.loc[data['currentSmoker'] == 1]['cigsPerDay']))
data['cigsPerDay'].fillna(cigsPerDay_mean, inplace=True)

totChol_mean = np.mean(data['totChol'])
data['totChol'].fillna(totChol_mean, inplace=True)

BMI_mean = np.mean(data['BMI'])
data['BMI'].fillna(BMI_mean, inplace=True)

glucose_mean = np.mean(data['glucose'])
data['glucose'].fillna(glucose_mean, inplace=True)

#One-out-of-K Encoding
#Turn DataFrame to NumPy ndArray
data_np = data.to_numpy()

#Create one-out-of-K Matrix
education = np.array(data_np[:, 2], dtype=int).T
K = education.max()+1
education_encoding = np.zeros((education.size, K))
education_encoding[np.arange(education.size), education] = 1
education_encoding = education_encoding[:,1:]

#Delete Column
data_np = np.delete(arr=data_np, obj=2, axis=1)

#Replace Deleted Columns
data_np = np.concatenate( (data_np[:, :], education_encoding), axis=1) 

#Creating the New DataFrame
cols = range(0, len(data.columns))
attributeNames = list(np.asarray(data.columns[cols]))
attributeNames.pop(2)
temp_col = ['educ1', 'educ2', 'educ3', 'educ4']
col = attributeNames + temp_col
data = pd.DataFrame(data_np, columns=col)

#DataFrame to NumPy Array
data_np = data.to_numpy()