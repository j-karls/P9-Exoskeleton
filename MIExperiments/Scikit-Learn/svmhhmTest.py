import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

dia = datasets.load_diabetes()
x, y = dia.data[:, np.newaxis, 2], dia.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
regr_obj = linear_model.LinearRegression()
regr_obj.fit(x_train, y_train)
y_predict = regr_obj.predict(x_test)

# print(regr_obj.coef_)
# print(regr_obj.intercept_)

plot.scatter(x_test, y_test, color = 'red')
plot.plot(x_test, regr_obj.predict(x_test), color = 'blue')
plot.title("Diabetes")
plot.xlabel("X")
plot.ylabel("Y")
plot.show()
