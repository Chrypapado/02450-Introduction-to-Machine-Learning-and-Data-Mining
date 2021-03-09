#Import script
from missing_values import *

#Turn DataFrame to NumPy ndArray
data_np = data.to_numpy()

#Perform one-out-of-K encoding to the categorical variables (not binary)
#Create one-out-of-K matrix
education = np.array(data_np[:, 2], dtype=int).T
K = education.max()+1
education_encoding = np.zeros((education.size, K))
education_encoding[np.arange(education.size), education] = 1
education_encoding = education_encoding[:,1:]

#Delete education column
data_np = np.delete(arr=data_np, obj=2, axis=1)

#Replace deleted column with the encoding
data_np = np.concatenate( (data_np[:, :], education_encoding), axis=1) 

#Split ndArray to X and y
X = np.delete(arr=data_np, obj=14, axis=1)
y = data_np[:, 14]

#Creating new dataframe
cols = range(0, len(data.columns) - 1)
attributeNames = list(np.asarray(data.columns[cols]))
attributeNames.pop(2)
temp_col = ['educ1', 'educ2', 'educ3', 'educ4']
col = attributeNames + temp_col
data = pd.DataFrame(X, columns=col)

#Number of data samples and attributes
N, M = X.shape

#Number of classes
C = 2