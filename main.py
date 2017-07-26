# tutorial at:
# https://www.youtube.com/watch?v=cKxRvEZd3Mw

from sklearn import tree

features = [[140, 1],[130, 1],[150, 2],[170, 2]]
labels = [1, 1, 2, 2];

classifier = tree.DecisionTreeClassifier()
#find pattern in trainning data
classifier = classifier.fit(features, labels)
#guess a new fruit
print (classifier.predict([[160, 1]]))

