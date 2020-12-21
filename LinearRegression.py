import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# Read and split data set to features and tagret
def readDataSet():
    house_dataset = pd.read_csv('USA_Housing.csv')

    features = house_dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                              'Avg. Area Number of Bedrooms', 'Area Population']]
    target = house_dataset['Price']
    return house_dataset, features, target


# plot all features with each other 
def plotFeatures(house_dataset):
    sns.pairplot(house_dataset)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# plot one feature with target
def plotOneFeatureWithTarget(features, target):
    plt.scatter(features['Avg. Area Income'], target)
    plt.xlabel('Avg. Area Income')
    plt.ylabel('Price')
    plt.show()


# add column of ones to get first weight
def addColumn(features):
    df = pd.DataFrame(features)
    features = df.to_numpy()  # change dataframe to numpy array
    one = np.ones(len(features))
    features = np.insert(features, 0, one, axis=1)
    return features


def predict(weights, features):
    prediction = features.dot(weights)  # multiply feature with weights
    return prediction


# Compute linear regression cost
def compute_cost(featuresTrain, targetTrain, weights):
    num_samples = len(featuresTrain)
    prediction = predict(weights, featuresTrain)
    errors = np.subtract(prediction, targetTrain)
    # errors.T.dot(errors) is equivalent to two steps summation and square
    cost = 1 / (2 * num_samples) * errors.T.dot(errors)
    return cost


def printCost(train, target, weights, iteration, cost):
    cost[iteration] = compute_cost(train, target, weights)
    print('--------------------------')
    print(f'iteration: {iteration + 1}')
    print(f'cost: {cost[iteration]}')


def gradientDescent(train, target, weights, alpha, max_iteration):
    iteration = 0
    numOfSamples = len(target)
    cost = np.zeros(max_iteration)  # make array of zeros to store cost of every iteration

    while iteration < max_iteration:
        printCost(train, target, weights, iteration, cost)
        prediction = predict(weights, train)
        errors = np.subtract(prediction, target)
        # calculate gradient
        gradient = (alpha / numOfSamples) * train.transpose().dot(errors)
        # subtract gradient to minimize cose
        weights = weights - gradient
        iteration += 1
    return weights, cost


# plot linear regression
def plotLine(featureTest, targetTest, weights):
    plt.figure()
    plt.scatter(featureTest, targetTest)
    plt.plot(featureTest, weights[0] + weights[1] * featureTest, 'r', label='Linear Regression')
    plt.show()


# plot curve of cost function
def plotCost(costList, max_iter):
    plt.plot(range(1, max_iter + 1), costList, color='red')
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.grid()
    plt.xlabel("Number of iterations")
    plt.ylabel("cost")
    plt.title("Convergence of gradient descent")
    plt.show()


# call al functions
def main():
    weights_0 = np.zeros(6)  # intilaize weight
    alpha = 1e-11 * 0.15
    max_iter = 1000
    houseDataSet, features, target = readDataSet()
    plotFeatures(houseDataSet)
    plotOneFeatureWithTarget(features, target)
    features_train, features_test, target_train, \
    target_test = train_test_split(features, target, test_size=0.2, random_state=1)
    features_train = addColumn(features_train)
    weights, costList = \
        gradientDescent(features_train, target_train, weights_0, alpha, max_iter)
    plotLine(features_test['Avg. Area Income'], target_test, weights)
    plotCost(costList, max_iter)
    print("weights:", weights)
    features_test = addColumn(features_test)
    print(compute_cost(features_test, target_test, weights))


main()
