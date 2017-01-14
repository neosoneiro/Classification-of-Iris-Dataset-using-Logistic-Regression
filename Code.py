import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

#Loading the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns=['sepallength','sepalwidth','petallength','petalwidth','class']
dataset=pandas.read_csv(url,names=columns)

#Summarizing the dataset
print dataset.shape
print dataset['class'].value_counts()
print "The various descriptions of the data are--"
print dataset.describe()

#Visualizing the data
#Univariate Visualization
print "Here are the histograms of the data "
dataset.hist(facecolor='red')
plt.show()

#Multivariate Visualization
print "Multivariate distribution of data "
scatter_matrix(dataset)
plt.show()

#visualization of various parameters of various species
print "Sepal LEngth and Sepal Width of various species is "
ax=dataset[dataset['class']=='Iris-setosa'].plot.scatter(x='sepallength',y='sepalwidth',color='red',label='setosa')
dataset[dataset['class']=='Iris-virginica'].plot.scatter(x='sepallength',y='sepalwidth',color='blue',label='virginica',ax=ax)
dataset[dataset['class']=='Iris-versicolor'].plot.scatter(x='sepallength',y='sepalwidth',color='green',label='versicolor',ax=ax)
ax.set_title("Scatter plot")
plt.show()


#Making the Model using different algorithms and evaluating various algorithms
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#print "Training X-"
#print x_train
#print "Training Y-"
#print y_train
model=LogisticRegression()
model.fit(x_train,y_train)
y_result=model.predict(x_test)
print "Accuracy of the model using Logistic Regression is "
print model.score(x_train,y_train)


