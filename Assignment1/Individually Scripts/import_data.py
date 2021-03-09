#Libraries
import pandas as pd 

#Import the dataset
data = pd.read_csv("framingham.csv")

#Settings
#pd.set_option('display.max_rows', None)
#pd.set_option('max_columns', 20) 

#Explore the data
print('Data head is printed below:')
print(data.head(), '\n')

print('Data info is printed below:')
print(data.info(), '\n')

print('The description of the data is printed below:')
print(data.describe(), '\n')