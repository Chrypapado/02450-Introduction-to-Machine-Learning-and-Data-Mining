#Libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Import script
from missing_values import *

#Preprocess
plot_titles = ['Age', 'Cigarettes Per Day', 'Total Cholesterol', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Body Mass Index', 'Heart Rate', 'Glucose']
cont_attributes = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
cont_data = data[cont_attributes]
filter_cigs = cont_data[cont_data['cigsPerDay'] != 0]

#Scatter plot
sns.pairplot(cont_data)
plt.show()