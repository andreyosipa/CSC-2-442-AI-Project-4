import numpy as np
import csv
import random
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
    
    def predict(self, X):
        return np.apply_along_axis(self.threshold,0,np.dot(np.mat(self.w), X.T) + self.b)
        
    def loss(self, X, y):
        #L2 loss
        return np.sum(np.abs(self.predict(X) - y)) + np.dot(self.w, self.w.T) + self.b**2
    
    def fit(self, X, y, X_dev=None, y_dev=None, n_iter=10000, plot=True):
        n_train = len(X)
        self.w = np.zeros(len(X[0]))
        self.b = 0
        time = 1
        finish = False
        idx = 0
        #list of accuracy values no dev set
        accuracy = []

        if X_dev==None:
            X_dev = X
            y_dev = y

        accuracy.append(1 - np.sum(np.abs(self.predict(X_dev) - y_dev))/len(X_dev))
        loss_prev = self.loss(X,y)
        while not finish:
            rate = 10.0 * self.lrate/ 10**(time//100 + 1)
            condition = y[idx] - self.threshold(np.dot(self.w,X[idx]) + self.b)
            self.w = self.w + rate*condition*X[idx]
            self.b = self.b + rate*condition   
            time += 1
            idx = (idx + 1) % n_train
            #add accuracy value only if change in weights happened
            if condition != 0:
                accuracy.append(1 - np.sum(np.abs(self.predict(X_dev) - y_dev))/len(X_dev))
            
            #check stoppping conditions
            if condition != 0 and time > n_iter/10:
                finish = abs(loss_prev - self.loss(X,y)) < 1e-5*self.loss(X,y)
                loss_prev = self.loss(X,y)
            
            if time > n_iter:
                break
        
        if plot:
            plt.figure(figsize=(20,10))
            plt.plot(accuracy)
            plt.xlabel('Weight updates')
            plt.ylabel('Accuracy')
            plt.title('Performance')
            plt.show()

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
    
    def fit(self, X, y, X_dev=None, y_dev=None, n_iter=10000, plot=True):
        n_train = len(X)
        self.w = np.zeros(len(X[0]))
        self.b = 0
        finish = False
        accuracy = []
        time = 1
        idx = 0

        if X_dev==None:
            X_dev = X
            y_dev = y

        loss_prev = self.loss(X,y)
        accuracy.append(1 - np.sum(np.abs(self.predict(X_dev) - y_dev))/len(X_dev))
        while not finish:
            rate = 2 * self.lrate/ 2**(time//100 + 1)
            self.w = self.w + rate*(y[idx] - self.h(X[idx]))*self.h(X[idx])*(1 - self.h(X[idx]))*X[idx]
            self.b = self.b + rate*(y[idx] - self.h(X[idx]))*self.h(X[idx])*(1 - self.h(X[idx]))  
            time += 1
            #randomly pick data entry to do update
            idx = random.randint(0, n_train-1)
            accuracy.append(1 - np.sum(np.abs(self.predict(X_dev) - y_dev))/len(X_dev))
            
            #stopping condition
            if time > n_iter:
                break
                
        if plot:
            plt.figure(figsize=(20,10))
            plt.plot(accuracy)
            plt.xlabel('Weight updates')
            plt.ylabel('Accuracy')
            plt.title('Performance')
            plt.show()

def crossValidation(classifier, X, y, folds=1, other_args={}):
    c = classifier()
    fold_size = int(len(X)/folds)
    accuracy = []
    for k in range(folds):
        #indexes in original data of entries in training set
        indexes = [x for x in range(len(X)) if x not in range(k*fold_size, (k+1)*fold_size)]
        c.fit(X[indexes], y[indexes], **other_args)
        accuracy.append((fold_size - sum(abs(c.predict(X[k*fold_size:(k+1)*fold_size]) - y[k*fold_size:(k+1)*fold_size])))/fold_size)
    print "accuracy: ", np.average(accuracy)*100, "%"

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

def earthquakeExampleLC(name='earthquake-noisy.data.txt', plot=False):
    data = np.loadtxt(name, delimiter=',')
    np.random.shuffle(data)
    c = LinearClassifier
    crossValidation(c, data[:,:2], data[:,2], folds=5, other_args={'plot':plot})

def adultDatasetExampleLC(name='adult_dataset/'):
    train = read_datafile(name+'a7a.train')
    test = read_datafile(name+'a7a.test')
    dev = read_datafile(name+'a7a.dev')
    c = LinearClassifier()
    c.fit(train[:,1:], (train[:,0] + 1)/2, X_dev=dev[:,1:], y_dev=(dev[:,0] + 1)/2)
    accuracy = (len(test) - sum(abs(c.predict(test[:,1:]) - (test[:,0] + 1)/2)))/len(test)
    print "accuracy: ", accuracy*100, "%"

def earthquakeExampleLR(name='earthquake-noisy.data.txt', plot=False):
    data = np.loadtxt(name, delimiter=',')
    np.random.shuffle(data)
    c = LogisticRegression
    crossValidation(c, data[:,:2], data[:,2], folds=5, other_args={'plot':plot, 'n_iter':500})

def adultDatasetExampleLR(name='adult_dataset/', plot=False):
    train = read_datafile(name+'a7a.train')
    test = read_datafile(name+'a7a.test')
    dev = read_datafile(name+'a7a.dev')
    c = LogisticRegression()
    c.fit(train[:,1:], (train[:,0] + 1)/2, X_dev=dev[:,1:], y_dev=(dev[:,0] + 1)/2, n_iter=200)
    accuracy = (len(test) - sum(abs(c.predict(test[:,1:]) - (test[:,0] + 1)/2)))/len(test)
    print "accuracy: ", accuracy*100, "%"