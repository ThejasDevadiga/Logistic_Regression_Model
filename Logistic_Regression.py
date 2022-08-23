# Logistic Regression ML-Model 
# Thejas Devadiga
# Date : 19/8/22


from Test_Model import Test_Model
from  Training_Model import Training_Model
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer



def main():
    X_tr = pd.read_csv('./train_X.csv')
    Y_tr= pd.read_csv('./train_Y.csv')
    X_te = pd.read_csv('./test_X.csv')
    Y_te = pd.read_csv('./test_Y.csv')
    
    X_train = X_tr.drop("Id" , axis=1).values
    Y_train = Y_tr.drop("Id" , axis=1).values
    X_test = X_te.drop("Id" , axis=1).values
    Y_test = Y_te.drop("Id" , axis=1).values
    learning_Rate = 0.005
    # X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.01,random_state=0)

    
    
    X_train = np.transpose(X_train)
    Y_train = Y_train.reshape(1,X_train.shape[1])
    
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # np.set_printoptions(precision=5)
    
    
    trained_Model = Training_Model(X_train,Y_train,learning_Rate)
    trained_Model.start(10000)
    print(trained_Model.RESULT[0])
    print(trained_Model.RESULT[1])
if __name__=='__main__':
    main()
