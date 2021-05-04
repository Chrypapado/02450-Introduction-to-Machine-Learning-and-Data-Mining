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

#Create the Classification Dataset
#Splitting the Predicted Value from the Attributes
idx_y = list(data.columns).index('TenYearCHD')
idx_X = list(range(0, idx_y)) + list(range(idx_y + 1, len(data.columns)))
X = data_np[:, idx_X]
y = data_np[:, idx_y]
attributeNames = list(data.columns[idx_X])
N, M = X.shape
C = 2
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
outer_ANN_error_train = []
k_hidden = np.empty((K1,1))
opt_k1_hidden = []
#Parameters for Neural Network Classifier
n_hidden_units = [1, 5, 10]      
n_replicates = 1        
max_iter = 10000

#==========================Baseline Linear Regression=========================#
baseline_error = np.zeros(K1)

#========================Regularized Logistic Regression========================#
rlr_train_error = np.zeros(K1)
rlr_test_error = np.zeros(K1)
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
    y_train_ANN1 = torch.tensor(y[train_index1], dtype=torch.float).view(-1,1)
    X_test_ANN1 = torch.tensor(X[test_index1,:], dtype=torch.float)
    y_test_ANN1 = torch.tensor(y[test_index1], dtype=torch.uint8).view(-1,1)
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
        y_train_ANN2 = torch.tensor(y[train_index2], dtype=torch.float).view(-1,1)
        X_test_ANN2 = torch.tensor(X[test_index2,:], dtype=torch.float)
        y_test_ANN2 = torch.tensor(y[test_index2], dtype=torch.float).view(-1,1)
        #Train the Net on Different Number of Hidden Units
        for i, number in enumerate(n_hidden_units):            
            #Define the Model
            model = lambda: torch.nn.Sequential(torch.nn.Linear(M, number), 
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(number, 1),
                                                torch.nn.Sigmoid())
            loss_fn = torch.nn.BCELoss()       
            #Train the Net on Training Data
            net, final_loss, learning_curve = train_neural_net(model, loss_fn, 
                                                               X=X_train_ANN2,
                                                               y=y_train_ANN2,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)    
            #Determine Estimated Class Labels for Test Set
            y_test_ANN2_sigmoid = net(X_test_ANN2)
            y_test_ANN2_pred = (y_test_ANN2_sigmoid > 0.5).to(torch.uint8)
            #Determine Errors and Errors
            y_test_inner = y_test_ANN2.type(dtype=torch.uint8)
            inner_e = y_test_ANN2_pred != y_test_inner
            inner_ANN_error_rate = (sum(inner_e).type(torch.float)/len(y_test_inner)).data.numpy()
            inner_ANN_error_hidden[k2,i] = inner_ANN_error_rate
        inner_ANN_error[k2] = np.min(np.mean(inner_ANN_error_hidden,axis=0))
        k_hidden[k2] = n_hidden_units[np.argmin(np.mean(inner_ANN_error_hidden,axis=0))] 
    
    #==========================Baseline Linear Regression=========================#
    #Initialize Variables
    class_1 = y_train1.sum()
    class_0 = len(y_train1) - class_1
    baseline_model = float(np.argmax([class_0, class_1]))
    baseline_error[k1] = np.sum(y_test1 != baseline_model) / len(y_test1)
    print('\nError rate (Baseline Linear Regression) {0}/{1}: {2}\n'.format(k1+1, K1, np.round(baseline_error[k1], decimals = 4)))   
    
    #========================Regularized Logistic Regression========================#     
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
            rlr_model = lm.LogisticRegression(penalty='l2', C=1/lambdas[l])
            rlr_model = rlr_model.fit(X_train2, y_train2)
            y_pred2 = rlr_model.predict(X_test2).T
            error_lambdas_k2[k2] = np.sum(y_pred2 != y_test2) / len(y_test2)
        error_lambdas_k1[l] = np.sum(error_lambdas_k2) / len(error_lambdas_k2)
    
    #=====================================ANN=====================================#    
    opt_k2_hidden = int(k_hidden[inner_ANN_error.argmin()].item())
    #Define the model
    model = lambda: torch.nn.Sequential(torch.nn.Linear(M, opt_k2_hidden), 
                                        torch.nn.Tanh(),   
                                        torch.nn.Linear(opt_k2_hidden, 1),
                                        torch.nn.Sigmoid())
    loss_fn = torch.nn.BCELoss() 
    #Train the Net on Training Data
    net, final_loss, learning_curve = train_neural_net(model, loss_fn,
                                                       X=X_train_ANN1,
                                                       y=y_train_ANN1,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    print('\nError Rate (ANN) {0}/{1}: {2:.3f}\n'.format(k1+1, K1, final_loss))
    lc.append(learning_curve)
    #Determine Estimated Class Labels for Test Set
    y_test_ANN1_sigmoid = net(X_test_ANN1)
    y_test_ANN1_pred = (y_test_ANN1_sigmoid > 0.5).to(torch.uint8)
    y_train_ANN1_sigmoid = net(X_train_ANN1)
    y_train_ANN1_pred = (y_train_ANN1_sigmoid > 0.5).to(torch.uint8)
    #Determine Errors and Errors
    y_train_outer = y_train_ANN1.type(dtype=torch.uint8)
    y_test_outer = y_test_ANN1.type(dtype=torch.uint8)
    outer_e = y_test_ANN1_pred != y_test_outer
    outer_error_rate = (sum(outer_e).type(torch.float)/len(y_test_outer)).data.numpy()
    outer_e_train = y_train_ANN1_pred != y_train_outer
    outer_error_rate_train = (sum(outer_e_train).type(torch.float)/len(y_train_outer)).data.numpy() 
    outer_ANN_error.append(outer_error_rate)
    outer_ANN_error_train.append(outer_error_rate_train)
    opt_k1_hidden.append(opt_k2_hidden)
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k1])
    h.set_label('CV fold {0}'.format(k1+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
    #========================Regularized Logistic Regression========================# 
    min_error = np.min(error_lambdas_k1)
    lambdas_opt_idx = np.argmin(error_lambdas_k1)
    lambdas_opt.append(lambdas[lambdas_opt_idx])
    rlr_model = lm.LogisticRegression(penalty='l2', C=1/lambdas[l])
    rlr_model = rlr_model.fit(X_train1, y_train1)
    y_train_pred1 = rlr_model.predict(X_train1).T
    y_pred1 = rlr_model.predict(X_test1).T
    rlr_train_error[k1] = np.sum(y_train_pred1 != y_train1) / len(y_train1)
    rlr_test_error[k1] = np.sum(y_pred1 != y_test1) / len(y_test1)
    print('\nError rate (Regularized Logistic Regression) {0}/{1}: {2}'.format(k1+1, K1, np.round(rlr_test_error[k1], decimals = 4)))
    print('Optimal lambda: {0}\n'.format(lambdas_opt[k1]))
    
#=====================================ANN=====================================#
# Display the error rate across folds
summaries_axes[1].bar(np.arange(1, K1+1), np.squeeze(np.asarray(outer_ANN_error)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K1+1))
summaries_axes[1].set_ylabel('Error rate')
summaries_axes[1].set_title('Test misclassification rates')
plt.savefig('Learning Curves Classification.png')

#Print Generalization Error
for i in range(K1): 
    print('Generalization Error of ANN for {0} Fold: \t{1:.3f}, \t Hidden Units: {2}'.format(i+1, *outer_ANN_error[i], opt_k1_hidden[i]))
    print('Generalization Error of RLR for {0} Fold: \t{1:.3f}, \t Lambda: {2}'.format(i+1, rlr_test_error[i], lambdas_opt[i]))
    print('Generalization Error of BLR for {0} Fold: \t{1:.3f}\n'.format(i+1, baseline_error[i]))

#Diagram of Best Neural Net in Last Fold
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

#Train and Test Error
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.bar(np.arange(10) - 0.4/2, np.squeeze(np.asarray(rlr_train_error)), 0.4, label='Train Error')
plt.bar(np.arange(10) + 0.4/2, np.squeeze(np.asarray(rlr_test_error)), 0.4, label='Test Error')
plt.xlabel('K-Fold')
plt.ylabel('Classification Error')
plt.title('Classification Error for Every K-Fold Validation (Logistic Regression)', fontsize=12)
plt.xticks(np.arange(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.legend()
plt.subplot(1,2,2)
plt.bar(np.arange(10) - 0.4/2, np.squeeze(np.asarray(outer_ANN_error_train)), 0.4, label='Train Error')
plt.bar(np.arange(10) + 0.4/2, np.squeeze(np.asarray(outer_ANN_error)), 0.4, label='Test Error')
plt.xlabel('K-Fold')
plt.ylabel('Classification Error')
plt.title('Classification Error for Every K-Fold Validation (ANN)', fontsize=12)
plt.xticks(np.arange(10), ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.legend()
plt.savefig('Classification Error for every K.png')
plt.show()

#Statistical Evaluation
alpha = 0.05
y_test_ANN = (y_test_ANN1_pred.data.numpy()).flatten()
#ANN/Regularized Logistic Regression
thetaANLO, ciANLO, p_ANLO = mcnemar(y_test1, y_test_ANN, y_pred1, alpha=alpha)
print('\n')
#ANN/Baseline
thetaANBA, ciANBA, p_ANBA = mcnemar(y_test1, y_test_ANN, baseline_model, alpha=alpha)
print('\n')
#Baseline/Regularized Logistic Regression
thetaBALO, ciBALO, p_BALO = mcnemar(y_test1, baseline_model, y_pred1, alpha=alpha)
print('\n')
#Print Results
print('(ANN/Baseline) Confidence Interval: [{0:.3f}, {1:.3f}] \t P-Value: {2}'.format(ciANBA[0], ciANBA[1], p_ANBA))
print('(ANN/RLR)      Confidence Interval: [{0:.3f}, {1:.3f}] \t P-Value: {2}'.format(ciANLO[0], ciANLO[1], p_ANLO))
print('(Baseline/RLR) Confidence Interval: [{0:.3f}, {1:.3f}] \t P-Value: {2}'.format(ciBALO[0], ciBALO[1], p_BALO))