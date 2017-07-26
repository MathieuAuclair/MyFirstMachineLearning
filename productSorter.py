import numpy as np
from sklearn import tree

# wich game is more apropriate for a 15 years old boy (0) wich is type 0

testInput = [15,0,0]

# age, gender, type (bool)
datasets = [[15,1,0],[16,0,1],[12,0,0],[20,1,1],[22,0,1],[21,1,1],[5,0,1],[6,0,0],[7,0,0]]
targets = [0,0,0,1,1,1,2,2,2]


#set tree
classifier = tree.DecisionTreeClassifier()

#find pattern in trainning data
classifier = classifier.fit(datasets, targets)

#guess a new entry
result = classifier.predict(testInput)

print (result)

