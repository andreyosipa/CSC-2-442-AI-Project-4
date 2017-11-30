import csv
import random
import numpy as np
import math
import matplotlib.pyplot as plt

def read_datafile(name, delimiter=','):
    with open(name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        data = []
        for row in reader:
            data.append(row)
    return data

def pluralityValue(data, col=-1):
    counters = {}
    for idx in range(len(data)):
        if data[idx][col] in counters.keys():
            counters[data[idx][col]] += 1
        else:
            counters[data[idx][col]] = 1
    max_count = 0
    for val in counters.keys():
        if counters[val] > max_count:
            max_val = val
            max_count = counters[val]
    return max_val

class Variable:
    def __init__(self, name, domain, idx):
        self.name = name
        self.domain = domain
        self.idx = idx
        
class Problem:
    def __init__(self):
        self.variables = []
    
    def add_variable(self, variable):
        self.variables.append(variable)
        
class DecisionTree:
    def __init__(self, variable=None, value=None):
        self.value = value
        self.variable = variable
        self.children = {}
        
    def dump(self, indent=0):
        if self.value != None:
            print ' '*indent + self.value
        else:
            print ' '*indent + self.variable.name
            for val in self.variable.domain:
                print ' '*indent + val
                self.children[val].dump(indent=indent+5)
    
    def classify(self, data):
        output = []
        for d in data:
            output.append(self.classifyOne(d))
        return output
            
    def classifyOne(self, data):
        if self.value != None:
            return self.value
        else:
            return self.children[data[self.variable.idx]].classifyOne(data)
            

def listWithout(lst, element):
    tmp = list(lst)
    tmp.remove(element)
    if tmp != None:
        return tmp
    return []
        
def learnTree(data, variables, parent_data):
    if len(data) == 0:
        return DecisionTree(value=pluralityValue(parent_data))
    if len(set([x[-1] for x in data])) == 1:
        return DecisionTree(value=data[0][-1])
    if len(variables) == 0:
        return DecisionTree(value=pluralityValue(data))
        
    #importance sampling
    feature = mostImportantFeature(data, variables)
        
    tree = DecisionTree(variable=feature)
    for val in feature.domain:
        #filter data
        exs = [x for x in data if x[feature.idx] == val]
        #create subtree
        subtree = learnTree(exs, listWithout(variables, feature), data)
        #add branch
        tree.children[val] = subtree
    return tree

def mostImportantFeature(data, variables):
    #method one vs all for each outcome
    splits_gain = {}
    outcomes = set([x[-1] for x in data])
    for var in variables:
        for out in outcomes:
            p,n = pnSamples(data, -1, out)
            remainder = 0
            for val in var.domain:
                pv,nv = pnSamples([x for x in data if x[var.idx] == val], -1, out)
                if pv+nv > 0:
                    remainder += float(pv+nv)/(p+n) * B(float(pv)/(pv+nv))
            splits_gain[(var,out)] = B(float(p)/(p+n)) - remainder
    
    print "gain for ", len(variables)
    for k in splits_gain.keys():
        print k[0].name, k[1], splits_gain[k]
    
    maxkeys = []
    for key in splits_gain.keys():
        if splits_gain[key] == max(splits_gain.values()):
            maxkeys.append(key[0])
    return maxkeys[random.randint(0, len(maxkeys) - 1)]
            
def B(x):
    if x == 1 or x == 0:
        return 0
    return -(x * math.log(x,2) + (1-x) * math.log(1-x,2))

def pnSamples(data, idx, pos_val):
    p = len([x for x in data if x[idx] == pos_val])
    n = len(data) - p
    return p,n

def willWaitExample():
    data = read_datafile('WillWait-data.txt')
    alt = Variable("Alternate", list(set([x[0] for x in data])), 0)
    bar = Variable("Bar", list(set([x[1] for x in data])), 1)
    fri = Variable("Fri/Sat", list(set([x[2] for x in data])), 2)
    hun = Variable("Hungry", list(set([x[3] for x in data])), 3)
    pat = Variable("Patrons", list(set([x[4] for x in data])), 4)
    pri = Variable("Price", list(set([x[5] for x in data])), 5)
    rai = Variable("Raining", list(set([x[6] for x in data])), 6)
    res = Variable("Reservation", list(set([x[7] for x in data])), 7)
    typ = Variable("Type", list(set([x[8] for x in data])), 8)
    wai = Variable("WaitEstimate", list(set([x[9] for x in data])), 9)
    T = learnTree(data, [alt, bar, fri, hun, pat, pri, rai, res, typ, wai], [])
    T.dump()

def irisDiscreteExample():
    data = read_datafile('iris.data.discrete.txt')
    sl = Variable("sepal length", list(set([x[0] for x in data])), 0)
    sw = Variable("sepal width", list(set([x[1] for x in data])), 1)
    pl = Variable("petal length", list(set([x[2] for x in data])), 2)
    pw = Variable("petal width", list(set([x[3] for x in data])), 3)

    train = data[:120]
    test = data[120:]

    T = learnTree(train, [sl, sw, pl, pw], [])

    T.dump()

    res = T.classify(test)
    correct = 0.0
    for idx in range(len(test)):
    if res[idx] == test[idx][-1]:
        correct += 1
    print "accuracy:", correct/len(test)