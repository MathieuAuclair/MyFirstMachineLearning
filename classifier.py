# tutorial at:
# https://www.youtube.com/watch?v=cKxRvEZd3Mw
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
test = [0,50,100]

#trainnig data
#we remove a valid entry from the list so we know it the 
#classifier is wrong since this info is suppose to be valid
trainTarget = np.delete(iris.target, test)
trainData = np.delete(iris.data, test, axis=0)

# testing the accuracy of our classifier
testTarget = iris.target[test]
testData = iris.data[test] 

classifier = tree.DecisionTreeClassifier()
#find pattern in trainning data
classifier = classifier.fit(testData, testTarget)

#guess a new entry
result = classifier.predict(testData)

print (result)

def printResult(classifierOutput):
    #Hard Code lvl 1.000.000!
    if(str(classifierOutput) == "[0 1 2]"):
        print ("classifier is right!")
    else:
        print ("classifier is wrong!")

printResult (result)
