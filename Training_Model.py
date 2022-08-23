
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import logsumexp
 
class Training_Model():
    def __init__(self,X_train,Y_train,learning_Rate):
        
        self.weight = np.zeros((X_train.shape[0],1))
        self.intercept = 0
        self.train_X = X_train
        self.train_Y = Y_train
        self.learning_Rate = learning_Rate
        self.regressor = Logistic_Regression(self.train_X,self.train_Y,self.intercept,self.learning_Rate,self.weight)
        self.RESULT = [[],[]]
    
    
    def start(self,steps):
        print("Starting training...")
        curve_points = self.regressor.sigmoid(self.train_X,self.weight,self.intercept)
        print("X :",self.train_X.shape)
        print("Y :",self.train_Y.shape)
        print("Weight :",self.weight.shape)
        # self.regressor.best_fit_plot(np.squeeze(self.train_X[2]),np.squeeze(self.train_Y[0]),np.squeeze(curve_points),"INITIAL FIT")
        iterations = 0
        while 1:
            self.regressor.fit_curve()
            iterations+=1
            if iterations % steps == 0:
                print("Iterations elapsed : ", iterations)
                print("Cost  : ",self.regressor.cost[-1] )
                print("ACCURACY: ", self.regressor.accuracy(self.train_X))
                stop = input("Do you want to stop trainingModel (y/*)??") 
                if stop =="y":
                    break
        self.weight = self.regressor.weights
        self.intercept = self.regressor.intercept
        self.regressor.get_Confusion_Matrix()
        plt.plot(range(iterations),self.regressor.cost)
        plt.show()
        result=self.regressor.get_Confusion_Matrix()
        sns.heatmap(result,linewidth=0.2)
        plt.show()

        
class Logistic_Regression():
    def __init__(self, train_X, train_Y,intercept,learning_Rate,weights):
        self.X = train_X
        self.Y = train_Y
        self.learning_Rate = learning_Rate
        self.weights = weights
        self.intercept = intercept
        self.m = train_X.shape[1]
        self.cost=[] 
        
        
    def sigmoid(self,X,Weight,Intercept):  
        Z =  np.dot(np.transpose(Weight),X)+Intercept
        return 1/(1+np.exp(-Z))
    

    
    def best_fit_plot(self,X,Y,Z,label):
        plt.figure(label)
        plt.scatter(X,Y,color='red')
        plt.plot(X,Z,color='green')
       
        plt.show()
        
    def cost_function(self,hypothesis):
        return  (-1/self.m)*np.sum(self.Y * np.log(hypothesis)+(1-self.Y)*np.log(1-hypothesis))
    
    def gradient_function(self, hypothesis):
        dW = (1/self.m)*np.dot(hypothesis-self.Y,np.transpose(self.X))
        dI = (1/self.m)*np.sum(hypothesis-self.Y)
        self.weights = self.weights - (self.learning_Rate * np.transpose(dW))
        self.intercept = self.intercept - (self.learning_Rate * (dI))
        return True
    
    def predict(self, test_X,weights,intercept ):
        result = self.sigmoid(test_X,weights,intercept )
        predicted_Y = np.zeros((result.shape))
        
        for i in range(len(result[0])):
            if result[0,i] >= 0.5:
                predicted_Y[0,i] = 1
            else:
                predicted_Y[0,i] =0
        
        return predicted_Y
        
    def fit_curve(self):
        sigma_Value = self.sigmoid(self.X,self.weights,self.intercept)
        self.cost.append(self.cost_function(sigma_Value))
        self.gradient_function(sigma_Value)
        return True
    
    def accuracy(self,test_X,):
        result = self.sigmoid(test_X,self.weights,self.intercept)
        result = result>=0.5
        res = np.array(result,dtype='int64')
        accuracy = (1-(np.sum(np.absolute(result -self.Y))/self.Y.shape[1]))*100
        return accuracy
    def get_Confusion_Matrix(self):
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0
        result= [[], [], [], []]
        predicted_Data = self.predict(self.X,self.weights,self.intercept)
        print("predicted_Data[i] = " , predicted_Data[0].shape)   
        print("Original Data: " , self.Y[0].shape) 
         
        for i in range(len(predicted_Data[0])): 
             
            if(predicted_Data[0][i]==1):
                if(predicted_Data[0][i]==self.Y[0][i]):
                    truePositive += 1
                else:
                    falsePositive += 1
            else:
                if(predicted_Data[0][i]==self.Y[0][i]):
                    trueNegative += 1
                else:
                    falseNegative += 1
        
        result[0].append(truePositive)
        result[1].append(falsePositive)
        result[2].append(trueNegative)
        result[3].append(falseNegative)
        return result
    
        
        
            
    