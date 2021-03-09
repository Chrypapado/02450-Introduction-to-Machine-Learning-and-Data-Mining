#Libraries
import matplotlib.pyplot as plt

#Import script
from data_preprocess import *

#Extracting the continuous attributes
plot_col = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
X_plot = data[plot_col]

#Standard Deviation Plot
range_ = np.arange(1, X_plot.shape[1] + 1)
plt.bar(range_, np.std(X_plot, axis = 0))
plt.xlabel('Attributes')
plt.ylabel('Standard Deviation')
plt.xticks(range_, plot_col, rotation=90)
plt.title('Attribute\'s Standard Deviations')

plt.show()