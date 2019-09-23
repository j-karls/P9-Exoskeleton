import matplotlib.pyplot as plot
import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import train_test_split


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def sort_two_lists_as_one(main_list, other_list):
    return (list(t) for t in zip(*sorted(zip(main_list, other_list))))
    # Todo: Need unittests to ensure this works


dia = datasets.load_diabetes()
x, y = dia.data[:, np.newaxis, 2], dia.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test, y_test = sort_two_lists_as_one(x_test, y_test)

# Linear Regression
lr_obj = linear_model.LinearRegression()
lr_obj.fit(x_train, y_train)
lr_predict = lr_obj.predict(x_test)

# Support-Vector-Regression
svr_obj = svm.SVR(kernel="poly", degree=2, C=10000, epsilon=0.01,
                  gamma=0.9, cache_size=10, verbose=True)
svr_obj.fit(x_train, y_train)
svr_predict = svr_obj.predict(x_test)

print("MAPE LR: " + str(mean_absolute_percentage_error(y_test, lr_predict)))
print("MAPE SVR: " + str(mean_absolute_percentage_error(y_test, svr_predict)))

plot.scatter(x_test, y_test, color="red", label="test set")
plot.plot(x_test, lr_predict, color="blue", label="linear")
plot.plot(x_test, svr_predict, color="yellow", label="svr")
plot.legend(loc='upper left')
plot.title("Diabetes")
# plot.xlabel("X")
# plot.ylabel("Y")
plot.show()
