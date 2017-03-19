import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.colors import LogNorm
from matplotlib import cm

data = np.loadtxt("ex2data1.txt", delimiter=',')
# data.columns = ['Population', 'Profit']


def sigmoid(hyp):
    return 1/(1+np.exp(-hyp))


def feature_normalization(X):
    X_mean_std = []

    for i in range(0, X.shape[0]):
        mean = X[i, :].mean()
        X[i, :] -= mean
        std = np.std(X[i, :])
        X[i, :] *= 1/std
        X_mean_std.append((mean, std))
    return X, np.array(X_mean_std)


def compute_cost(X, y, theta):
    m = y.shape[0]
    hypothesis = sigmoid(np.dot(X.transpose(), theta))
    loss = hypothesis.transpose() - y
    cost = np.sum(loss)/(2*m)
    return cost


def gradient_descent(X, y, theta, m, n_iter, alpha):
    cost_list = []

    for i in range(0, n_iter):
        hypothesis = np.dot(X.transpose(), theta)
        loss = hypothesis.transpose() - y
        cost = np.sum(loss**2)/(2*m)
        cost_list.append((i, cost))
        if i == 0:
            cost_0 = cost
        # print "Iteration {} | Cost: {}".format(i, cost)
        gradient = np.dot(X, loss.transpose())/m
        theta = theta - alpha * gradient
    print "Iteration {} | Cost: {}".format(i, cost)
    print "Cost Reduced by: {}".format(cost_0 - cost)
    return theta, np.array(cost_list)


def cost_surface_plot(X, y, theta):

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


def cost_contour_plot(X, y, theta):

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

    ax.contourf(a, b, J)
    # levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)

    plt.show()


def plot_fig(ax, x, y, label):

    return ax.plot(x, y, label=label)


def learning_rate_trials(X, y, theta, n, iterations, rates_to_try=[0.01, 0.02, 0.05, 0.001], plot=False):
    cost_matrix = []

    for rate in rates_to_try:
        theta, cost_decay = gradient_descent(X, y, theta, n, iterations, rate)
        cost_matrix.append(cost_decay[:, -1])

    print np.array(cost_matrix).shape
    cost_matrix = np.vstack((np.arange(iterations), np.array(cost_matrix)))

    if plot:
        fig, ax = plt.subplots(1, 1)
        x = cost_matrix[0, :]
        for i in range(1, cost_matrix.shape[0]):
            label = rates_to_try[i-1]
            plot_fig(ax, x, cost_matrix[i, :], label=label)


        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost_decay')
        plt.show()
    return cost_matrix


theta = np.zeros((3, 1))
n = data.shape[0]
X = np.vstack([data[:, 0], data[:, 1]])
print X.shape
# print X[:3, :3]
y = data[:, -1]
# print y[:3]

iterations = 1000
alpha = 0.01

X_norm, X_mean_std = feature_normalization(X)

X = np.vstack([np.ones(n), X_norm])
#
# y_line = np.dot(X.transpose(), theta)
#
# cost_surface_plot(X, y, theta)
#
# cost_contour_plot(X, y, theta)

Cost_Mat = learning_rate_trials(X, y, theta, n, 500, plot=True)

theta, cost_list = gradient_descent(X, y, theta, n, iterations, alpha)
