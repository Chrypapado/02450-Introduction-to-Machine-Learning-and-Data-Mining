#Libraries
import matplotlib.pyplot as plt
from scipy.linalg import svd

#Import script
from data_preprocess import *

#Standardization
Y1 = X.copy()
Y2 = X.copy()

#Subtract Mean from the Data
Y1[:,1] = X[:,1] - np.concatenate(np.ones(shape = (N, 1))*X[:,1].mean(axis=0)).ravel()
Y1[:,3] = X[:,3] - np.concatenate(np.ones(shape = (N, 1))*X[:,3].mean(axis=0)).ravel()
Y1[:,8:14] = X[:,8:14] - np.ones(shape = (N, 1))*X[:,8:14].mean(axis=0)

#Subtract Mean from the Data and Divide by Standard Deviation
Y2[:,1] = X[:,1] - np.concatenate(np.ones(shape = (N, 1))*X[:,1].mean(axis=0)).ravel()
Y2[:,3] = X[:,3] - np.concatenate(np.ones(shape = (N, 1))*X[:,3].mean(axis=0)).ravel()
Y2[:,8:14] = X[:,8:14] - np.ones(shape = (N, 1))*X[:,8:14].mean(axis=0)
Y2[:,1] = Y2[:,1]*(1/np.std(Y2[:,1], 0))
Y2[:,3] = Y2[:,3]*(1/np.std(Y2[:,3], 0))
Y2[:,8:14] = Y2[:,8:14]*(1/np.std(Y2[:,8:14], 0))

#Store the Two in a Cell, so we can just Loop over them
Ys = [Y1, Y2]

titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9

i = 0
j = 1

#Settings
plt.figure(figsize=(14,6))
plt.subplots_adjust(hspace=.7)
plt.subplots_adjust(wspace=.4)
nrows=3
ncols=2

for k in range(2):
    U,S,V = svd(Ys[k], full_matrices=False)
    V=V.T  
    if k==1: V = -V; U = -U
    rho = (S*S) / (S*S).sum() 
    Z = U*S

    #Plot Projection
    plt.subplot(nrows, ncols, 1+k)  
    for c in range(C):
        plt.plot(Z[y==c,i], Z[y==c,j], 'o', alpha = 0.7)
        
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))    
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(['No risk of CHD', 'Risk of CHD'])
    plt.axis('equal')

    #Plot Attribute Coefficients
    plt.subplot(nrows, ncols,  3+k)  
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], col[att])
        
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    
    #Add a Unit Circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));   
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')

    #Plot Cumulative Variance explained
    plt.subplot(nrows, ncols,  5+k)
    
    plt.plot(range(1,len(rho)+1), rho, 'x-')
    plt.plot(range(1,len(rho)+1), np.cumsum(rho), 'o-')
    plt.plot([1,len(rho)], [threshold, threshold], 'k--')
    
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')

plt.show()