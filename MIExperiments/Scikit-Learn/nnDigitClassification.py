import matplotlib.pyplot as plot
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

def display_plot(plot_obj, x, y, y_predict, title, xlabel, ylabel):
    plot_obj.scatter(x, y)
    plot_obj.plot(x, y_predict)
    # plot_obj.grid()
    plot_obj.title(title)
    plot_obj.xlabel(xlabel)
    plot_obj.ylabel(ylabel)
    plot_obj.show()

dia = datasets.load_diabetes()
x, y = dia.data, dia.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
regr_obj = linear_model.LinearRegression()
regr_obj.fit(x_train, y_train)
y_predict = regr_obj.predict(x_test)

print(len(x_test))
print(len(y_test))

# Todo: Why in the fuck is x_test a two-dimensional array? What is the second dimension used for?

display_plot(plot, x_test[:,0], y_test, y_predict, "Min vs Max", "MinTemp", "MaxTemp")

# todo why is it not showing my predicted line?
