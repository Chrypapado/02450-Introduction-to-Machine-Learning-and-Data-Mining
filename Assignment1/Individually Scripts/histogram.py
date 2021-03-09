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

#Histogram
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,20), sharex=False)

for i, att in enumerate(cont_attributes):
    if att == 'cigsPerDay':
        sns.histplot(x=filter_cigs[att], ax=axes[i//2,i%2]).set_title(plot_titles[i]);
        axes[i//2,i%2].set(xlabel=None)
        axes[i//2,i%2].set(ylabel=None)
    else:
        sns.histplot(x=cont_data[att], ax=axes[i//2,i%2]).set_title(plot_titles[i]);
        axes[i//2,i%2].set(xlabel=None)
        axes[i//2,i%2].set(ylabel=None)
        
plt.show()