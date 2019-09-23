import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import train_test_split

dia = datasets.load_diabetes()
x, y = dia.data[:, np.newaxis, 2], dia.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Linear Regression
lr_obj = linear_model.LinearRegression()
lr_obj.fit(x_train, y_train)
lr_predict = lr_obj.predict(x_test)

# Support-Vector-Regression
svr_obj = svm.SVR(kernel="poly", degree=1, C=1000, gamma="auto", verbose=True)
svr_obj.fit(x_train, y_train)
svr_predict = svr_obj.predict(x_test)

plot.scatter(x_test, y_test, color="red", label="test set")
plot.plot(x_test, lr_predict, color="blue", label="linear")
plot.plot(x_test, svr_predict, color="yellow", label="svr")
plot.legend(loc='upper left')
plot.title("Diabetes")
# plot.xlabel("X")
# plot.ylabel("Y")
plot.show()
