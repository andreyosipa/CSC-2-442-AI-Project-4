import numpy as np
import csv

import matplotlib.pyplot as plt

class LinearClassifier:
    def __init__(self, lrate=1.0):
        self.w = 0
        self.b = 0
        self.lrate = lrate
    
    def threshold(self, x):
        if x >= 0:
            return 1
        return 0
    
    def predict(self,X):
        return np.apply_along_axis(self.threshold,0,np.dot(np.mat(self.w), X.T) + self.b)
        
    def loss(self, X, y):
        return np.sum(np.abs(self.predict(X) - y))/len(X) + np.dot(self.w, self.w.T) + self.b**2
    
    def fit(self,X,y):
        n_train = len(X)
        self.w = np.zeros(len(X[0]))
        self.b = 0
        time = 1
        finish = False
        idx = 0
        accuracy = []
        accuracy.append(1 - np.sum(np.abs(self.predict(X) - y))/len(X))
        loss_prev = self.loss(X,y)
        while not finish:
            rate = 10.0 * self.lrate/ 10**(time//100 + 1)
            condition = y[idx] - self.threshold(np.dot(self.w,X[idx]) + self.b)
            self.w = self.w + rate*condition*X[idx]
            self.b = self.b + rate*condition   
            time += 1
            idx = (idx + 1) % n_train
            if condition != 0:
                accuracy.append(1 - np.sum(np.abs(self.predict(X) - y))/len(X))
            
            if condition != 0 and time > 1000:
                finish = abs(loss_prev - self.loss(X,y)) < 1e-3*self.loss(X,y)
                loss_prev = self.loss(X,y)
            
            if time > 10000:
                break
        plt.plot(accuracy)

class LogisticRegression:
    def __init__(self, rate=1.0):
        self.w = 0
        self.b = 0
        self.lrate = rate
    
    def h(self, x):
        return 1.0/(1 + np.exp(-np.dot(self.w, x) - self.b))
    
    def threshold(self, x):
        if x >= 0:
            return 1
        return 0
    
    def loss(self, X, y):
        return np.sum(np.abs(self.predict(X) - y))/len(X) + np.dot(self.w, self.w.T) + self.b**2
    
    def predict(self,X):
        return np.apply_along_axis(self.threshold,0,np.dot(np.mat(self.w), X.T) + self.b)
    
    def fit(self, X, y, n_iter=10000):
        n_train = len(X)
        self.w = np.zeros(len(X[0]))
        self.b = 0
        finish = False
        accuracy = []
        time = 1
        idx = 0
        loss_prev = self.loss(X,y)
        accuracy.append(1 - np.sum(np.abs(self.predict(X) - y))/len(X))
        while not finish:
            rate = 2 * self.lrate/ 2**(time//100 + 1)
            self.w = self.w + rate*(y[idx] - self.h(X[idx]))*self.h(X[idx])*(1 - self.h(X[idx]))*X[idx]
            self.b = self.b + rate*(y[idx] - self.h(X[idx]))*self.h(X[idx])*(1 - self.h(X[idx]))  
            time += 1
            idx = random.randint(0, n_train-1)
            accuracy.append(1 - np.sum(np.abs(self.predict(X) - y))/len(X))
            
            if time > n_iter:
                break
                
        #print accuracy
        plt.plot(accuracy)

def read_datafile(name):
    #Technical function to read data from files.
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        num = len(list(reader))
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data = np.zeros((num,124))
        idx = 0
        for row in reader:
            data[idx,0] = int(row[0])
            data[idx, [int(x.split(":")[0]) for x in row[1:-1]]] = 1
            idx += 1
    return data

def earthquakeExample(name='earthquake-noisy.data.txt'):
	data = np.loadtxt(name, delimiter=',')
	n_train = round(len(data) * 0.8)
	c = LinearClassifier()
	random.shuffle(data)
	train = data[:n_train,:]
	test = data[n_train:,:]
	c.fit(train[:,:2], train[:,2])

	accuracy = (len(test) - sum(abs(c.predict(test[:,:2]) - test[:,2])))/len(test)

	print "accuracy: ", accuracy, "%"

def adultDatasetExample(name='adult_dataset/'):
	train = read_datafile(name+'a7a.train')
	test = read_datafile(name+'a7a.test')
	dev = read_datafile(name+'a7a.dev')
	
	c = LinearClassifier()
	c.fit(data[:,1:], (data[:,0] + 1)/2)

	accuracy = (len(test) - sum(abs(c.predict(test[:,:2]) - (test[:,2] + 1)/2)))/len(test)

	print "accuracy: ", accuracy, "%"	