#from pylab import *
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals.six import StringIO

rawData = loadmat("Data/wine.mat")


X = rawData["X"]
y = rawData["y"]
attrName = rawData["attributeNames"][0]
className = rawData["classNames"][0]

clf = RandomForestClassifier(min_samples_split=100)
clf.fit(X, y)
print(clf.score(X,y))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(min_samples_split=20)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))

