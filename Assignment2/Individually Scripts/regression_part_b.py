#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.metrics import mean_squared_error,mean_absolute_error  
import torch

#Import Scripts
from functions import *
from data_preprocess import *

#Create the Regression Dataset
#Create new DataFrame with Selected Features
data_fs = data[['male', 'age', 'BPMeds', 'prevalentHyp', 'sysBP', 'diaBP', 'heartRate', 'glucose', 'TenYearCHD', 'educ4']]
#DataFrame to NumPy Array
data_np_fs = data_fs.to_numpy()
#Splitting the Predicted Value from the Attributes
idx_y = list(data_fs.columns).index('sysBP')
idx_X = list(range(0, idx_y)) + list(range(idx_y + 1, len(data_fs.columns)))
X = data_np_fs[:, idx_X]
y = data_np_fs[:, idx_y]
attributeNames = list(data_fs.columns[idx_X])
N, M = X.shape
#Normalize the Data
X = stats.zscore(X)

#K-Fold Cross Validation
K1 = 10
CV1 = model_selection.KFold(n_splits=K1,shuffle=True, random_state=0)
K2 = 10
CV2 = model_selection.KFold(n_splits=K2,shuffle=True, random_state=0)

#Initialize Variables
#=====================================ANN=====================================#
inner_ANN_error = np.empty((K1,1))
outer_ANN_error = []
k_hidden = np.empty((K1,1))
opt_k1_hidden = []
#Parameters for Neural Network Classifier
n_hidden_units = [1, 3, 5]      
n_replicates = 1        
max_iter = 10000

#==========================Baseline Linear Regression=========================#
baseline_error = np.zeros(K1)

#========================Regularized Linear Regression========================#
rlr_error = np.zeros(K1)
lambdas_opt = []

#Training
#=====================================ANN=====================================#
#Setup Figure for Display of Learning Curves and Error Rates in Fold
lc = []
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
color_list = ['tab:orange', 'tab:green', 'tab:purple', 
              'tab:brown', 'tab:pink', 'tab:gray', 
              'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
for (k1, (train_index1, test_index1)) in enumerate(CV1.split(X,y)): 
    print('\nOuter Crossvalidation fold: {0}/{1}'.format(k1+1,K1))
    #Extract Training and Test set for Current CV Fold
    X_train1 = X[train_index1,:]
    y_train1 = y[train_index1]
    X_test1 = X[test_index1,:]
    y_test1 = y[test_index1]

    #=====================================ANN=====================================#
    #Convert to Tensors
    X_train_ANN1 = torch.tensor(X[train_index1,:], dtype=torch.float)
    y_train_ANN1 = torch.tensor(y[train_index1], dtype=torch.float)
    X_test_ANN1 = torch.tensor(X[test_index1,:], dtype=torch.float)
    y_test_ANN1 = torch.tensor(y[test_index1], dtype=torch.uint8)
    #Initialize Variables
    inner_ANN_error_hidden = np.empty((K2,len(n_hidden_units)))
    for (k2, (train_index2, test_index2)) in enumerate(CV2.split(X_train1, y_train1)):
        print('\nInner Crossvalidation fold: {0}/{1}'.format(k2+1,K2))
        #Extract Training and Test set for Current CV Fold
        X_train2 = X[train_index2,:]
        y_train2 = y[train_index2]
        X_test2 = X[test_index2,:]
        y_test2 = y[test_index2]
        #Convert to Tensors
        X_train_ANN2 = torch.tensor(X[train_index2,:], dtype=torch.float)
        y_train_ANN2 = torch.tensor(y[train_index2], dtype=torch.float)
        X_test_ANN2 = torch.tensor(X[test_index2,:], dtype=torch.float)
        y_test_ANN2 = torch.tensor(y[test_index2], dtype=torch.uint8)
        #Train the Net on Different Number of Hidden Units
        for i, number in enumerate(n_hidden_units):
            #Define the Model
            model = lambda: torch.nn.Sequential(torch.nn.Linear(M, number), 
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(number, 1))
            loss_fn = torch.nn.MSELoss() 
            #Train the Net on Training Data
            net, final_loss, learning_curve = train_neural_net(model, loss_fn, 
                                                               X=X_train_ANN2,
                                                               y=y_train_ANN2,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            #Determine Estimated Class Labels for Test Set
            y_test_ANN2_pred = net(X_test_ANN2)
            #Turn to NumPy
            y_test_ANN2_np = y_test_ANN2.type(torch.float).data.numpy()
            y_test_ANN2_pred_np = y_test_ANN2_pred.type(torch.float).data.numpy().reshape((y_test_ANN2.shape[0],))
            #Determine Errors and Errors
            inner_ANN_error_hidden[k2,i] = np.square(y_test_ANN2_np-y_test_ANN2_pred_np).sum(axis=0)/y_test_ANN2_np.shape[0]
        inner_ANN_error[k2] = np.min(np.mean(inner_ANN_error_hidden,axis=0))
        k_hidden[k2] = n_hidden_units[np.argmin(np.mean(inner_ANN_error_hidden,axis=0))] 
    
    #==========================Baseline Linear Regression=========================#
    baseline_error[k1] = np.square(y_test1 - y_test1.mean()).sum() / len(y_test1)
    print('\nError rate (Baseline Linear Regression) {0}/{1}: {2}\n'.format(k1+1, K1, np.round(baseline_error[k1], decimals = 2)))   
    
    #========================Regularized Linear Regression========================#     
    #Initialize Variables
    lambdas = np.power(10.,range(-5,9))
    error_lambdas_k1 = np.zeros(len(lambdas))
    for l in range(len(lambdas)):
        error_lambdas_k2 = np.zeros(K2)
        for (k2 , (train_index2, test_index2)) in enumerate(CV2.split(X_train1, y_train1)):
            #Extract Training and Test Set for Current CV Fold
            X_train2 = X[train_index2,:]
            y_train2 = y[train_index2]
            X_test2 = X[test_index2,:]
            y_test2 = y[test_index2]
            rlr_model = lm.Ridge(alpha = lambdas[l], fit_intercept = True)
            rlr_model = rlr_model.fit(X_train2, y_train2)
            y_pred2 = rlr_model.predict(X_test2).T
            error_lambdas_k2[k2] = np.square(y_test2 - y_pred2).sum() / len(y_pred2)
        error_lambdas_k1[l] = np.sum(error_lambdas_k2) / len(error_lambdas_k2)
    
    #=====================================ANN=====================================#    
    opt_k2_hidden = int(k_hidden[inner_ANN_error.argmin()].item())
    #Define the model
    model = lambda: torch.nn.Sequential(torch.nn.Linear(M, opt_k2_hidden), 
                                        torch.nn.Tanh(),   
                                        torch.nn.Linear(opt_k2_hidden, 1))
    loss_fn = torch.nn.MSELoss() 
    #Train the Net on Training Data
    net, final_loss, learning_curve = train_neural_net(model, loss_fn,
                                                       X=X_train_ANN1,
                                                       y=y_train_ANN1,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    print('\nError rate (ANN) {0}/{1}: {2}\n'.format(k1+1, K1, final_loss))
    lc.append(learning_curve)
    #Determine Estimated Class Labels for Test Set
    y_test_ANN1_pred = net(X_test_ANN1)
    #Turn to NumPy
    y_test_ANN1_np = y_test_ANN1.type(torch.float).data.numpy()
    y_test_ANN1_pred_np = y_test_ANN1_pred.type(torch.float).data.numpy().reshape((y_test_ANN1.shape[0],))
    #Determine Errors and Errors
    mse = np.square(y_test_ANN1_np-y_test_ANN1_pred_np).sum(axis=0)/y_test_ANN1_np.shape[0]
    outer_ANN_error.append(mse) 
    opt_k1_hidden.append(opt_k2_hidden)
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k1])
    h.set_label('CV fold {0}'.format(k1+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
    #========================Regularized Linear Regression========================# 
    min_error = np.min(error_lambdas_k1)
    lambdas_opt_idx = np.argmin(error_lambdas_k1)
    lambdas_opt.append(lambdas[lambdas_opt_idx])
    rlr_model = lm.Ridge(alpha = lambdas[lambdas_opt_idx], fit_intercept = True)
    rlr_model = rlr_model.fit(X_train1, y_train1)
    y_pred1 = rlr_model.predict(X_test1).T
    rlr_error[k1] = np.square(y_test1 - y_pred1).sum() / len(y_test1)
    print('\nError rate (Regularized Linear Regression) {0}/{1}: {2}'.format(k1+1, K1, np.round(rlr_error[k1], decimals = 2)))
    print('Optimal lambda: {0}\n'.format(lambdas_opt[k1]))
    
#=====================================ANN=====================================#
# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K1+1), np.squeeze(np.asarray(outer_ANN_error)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K1+1))
summaries_axes[1].set_ylabel('MSE');
summaries_axes[1].set_title('Test mean-squared-error')
plt.savefig('Learning Curves.png')

#Print Generilazation Error
for i in range(K1): 
    print('Generalization Error of ANN for {0} Fold: \t{1:.2f}, \t Hidden Units: {2}'.format(i+1, outer_ANN_error[i], opt_k1_hidden[i]))
    print('Generalization Error of RLR for {0} Fold: \t{1:.2f}, \t Lambda: {2}'.format(i+1, rlr_error[i], lambdas_opt[i]))
    print('Generalization Error of BLR for {0} Fold: \t{1:.2f}\n'.format(i+1, baseline_error[i]))

#Diagram of Best Neural Net in Last Fold
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

#Estimated vs True Value
#========================Regularized Linear Regression========================# 
y_true = y_test1
y_pred = y_pred1
plt.figure(figsize=(10,10))
axis_range = [np.min([y_true, y_pred])-1, np.max([y_true, y_pred])+1]
plt.plot(axis_range, axis_range, 'k--')
plt.plot(y_true, y_pred,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Predicted vs True values')
plt.xlabel('True value')
plt.xlim(axis_range)
plt.ylabel('Predicted value')
plt.ylim(axis_range) 
plt.grid()
plt.savefig('Predicted vs True Values RLR')

#=====================================ANN=====================================#
y_true = y_test_ANN1_np
y_pred = y_test_ANN1_pred_np
plt.figure(figsize=(10,10))
axis_range = [np.min([y_true, y_pred])-1, np.max([y_true, y_pred])+1]
plt.plot(axis_range, axis_range, 'k--')
plt.plot(y_true, y_pred,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Predicted vs True values')
plt.xlabel('True value')
plt.xlim(axis_range)
plt.ylabel('Predicted value')
plt.ylim(axis_range) 
plt.grid()
plt.savefig('Predicted vs True Values ANN')
plt.show()

#Statistical Evaluation
#Initialize Variables
alpha = 0.05
length = len(y_test1)
y_true = np.array(y_test1).reshape(length,1)
y_rlr = np.array(y_pred1).reshape(length,1)
y_ann = list(float(i) for i in y_test_ANN1_pred_np)
y_ann = np.array(y_ann).reshape(length,1)

#Compute z and Confidence Interval of ANN
zANN = np.abs(y_true - y_ann) ** 2 
ciANN = stats.t.interval(1-alpha, df=len(zANN)-1, loc=np.mean(zANN), scale=stats.sem(zANN))  

#Compute z and Confidence Interval of BLR
zBASE = np.abs(y_true - y_test1.mean()) ** 2
ciBASE = stats.t.interval(1-alpha, df=len(zBASE)-1, loc=np.mean(zBASE), scale=stats.sem(zBASE)) 

#Compute z and Confidence Interval of RLR
zLIN = np.abs(y_true - y_rlr) ** 2
ciLIN = stats.t.interval(1-alpha, df=len(zLIN)-1, loc=np.mean(zLIN), scale=stats.sem(zLIN)) 

#Compare ANN and Baseline
zANBA = zBASE - zANN
ciANBA = stats.t.interval(1-alpha, len(zANBA)-1, loc=np.mean(zANBA), scale=stats.sem(zANBA))  
p_ANBA = stats.t.cdf( -np.abs( np.mean(zANBA) )/stats.sem(zANBA), df=len(zANBA)-1) 

#Compare ANN and RLR
zANLI = zANN - zLIN
ciANLI = stats.t.interval(1-alpha, len(zANLI)-1, loc=np.mean(zANLI), scale=stats.sem(zANLI))  
p_ANLI = stats.t.cdf( -np.abs( np.mean(zANLI) )/stats.sem(zANLI), df=len(zANLI)-1)

#Compare bASELINE and RLR
zBALI = zBASE - zLIN
ciBALI = stats.t.interval(1-alpha, len(zBALI)-1, loc=np.mean(zBALI), scale=stats.sem(zBALI))  
p_BALI = stats.t.cdf( -np.abs( np.mean(zBALI) )/stats.sem(zBALI), df=len(zBALI)-1) 

#Print Results
print('(ANN/Baseline) Confidence Interval: [{0:.3f}, {1:.3f}] \t P-Value: {2}'.format(*ciANBA[0], *ciANBA[1], *p_ANBA))
print('(ANN/RLR)      Confidence Interval: [{0:.3f}, {1:.3f}] \t P-Value: {2}'.format(*ciANLI[0], *ciANLI[1], *p_ANLI))
print('(Baseline/RLR) Confidence Interval: [{0:.3f}, {1:.3f}] \t\t P-Value: {2}'.format(*ciBALI[0], *ciBALI[1], *p_BALI))