import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import LogNorm
from matplotlib import cm

dir_ = "/Users/Jaspreet/Documents/Python_Workspace/ML_Courses/ML_Andrew_Ng/machine-learning-ex1/ex1/"

data = np.loadtxt(dir_+ "ex1data1.txt", delimiter=',')
# data.columns = ['Population', 'Profit']

print data.shape

theta = np.zeros((2,1))
n = data.shape[0]
X = np.vstack([np.ones(n),data[:,0]])
y = data[:,1]

iterations = 5000
alpha = 0.01


def compute_cost(X, y, theta):
    m = y.shape[0]
    hypothesis = np.dot(X.transpose(), theta)
    loss = hypothesis.transpose() - y
    cost = np.sum(loss**2)/(2*m)
    return cost


def gradient_descent(X, y, theta, m, n_iter, alpha):

    for i in range(0, n_iter):
        hypothesis = np.dot(X.transpose(), theta)
        loss = hypothesis.transpose() - y
        cost = np.sum(loss**2)/(2*m)
        # print "Iteration {} | Cost: {}".format(i, cost)
        gradient = np.dot(X, loss.transpose())/m
        theta = theta - alpha * gradient
    print "Iteration {} | Cost: {}".format(i, cost)
    print theta
    return theta


def cost_plot(X, y, theta, step):

    xmin, xmax = -10, 10
    ymin, ymax = -5, 5
    # print xmin
    theta_0 = np.arange(xmin, xmax, 0.5)
    theta_1 = np.arange(ymin, ymax, 0.25)
    a, b = np.meshgrid(theta_0, theta_1)

    J = np.array([compute_cost(X, y, np.array([a_x,b_x]).reshape(2,1)) for a_x, b_x in zip(theta_0, theta_1)])
    # print np.amin(J)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot_surface(a, b, J, norm=LogNorm(), rstride=1, cstride=1,
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    ax.plot(theta[1],theta[0], np.amin(J), 'r*', markersize=10)

    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_zlabel('cost')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    plt.show()

theta = gradient_descent(X, y, theta, n, iterations, alpha)

y_line = np.dot(X.transpose(), theta)

cost_plot(X, y, theta, 0.01)


# plt.scatter(data[:,0], data[:,1])
# plt.plot(data[:,0], y_line, 'r')
# plt.show()
