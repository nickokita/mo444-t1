import csv
import sys
import numpy
import matplotlib.pyplot as plt
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold

def convert_nums(diamond):
    cut_table = {
            'Fair': 1,
            'Good': 2,
            'Very Good': 3,
            'Premium': 4,
            'Ideal': 5
    }

    color_table = {
            "D": 7,
            "E": 6,
            "F": 5,
            "G": 4,
            "H": 3,
            "I": 2,
            "J": 1
    }

    clarity_table = {
            "I3": 1,
            "I2": 2,
            "I1": 3,
            "SI2": 4,
            "SI1": 5,
            "VS2": 6,
            "VS1": 7,
            "VVS2": 8,
            "VVS1": 9,
            "IF": 10,
            "FL": 11
    }

    diamond[1] = cut_table[diamond[1]]
    diamond[2] = color_table[diamond[2]]
    diamond[3] = clarity_table[diamond[3]]

def pol2(diamond):
    _size = len(diamond)-2
    for i in range(0,_size):
        for j in range(i,_size):
            diamond.append(float(diamond[i])*float(diamond[j]))
    return diamond

def pol3(diamond):
    _size = len(diamond)-1
    for i in range(0,_size):
        for j in range(i+9,_size):
            if i != 9 and j != 9:
                diamond.append(float(diamond[i])*float(diamond[j]))
    return diamond

def load_data(input_diamonds):
    f_diamonds = open(input_diamonds, newline='')
    d_reader = csv.reader(f_diamonds, delimiter=',')
    d_list = list(d_reader)
    d_list.pop(0)

    for diamond in d_list:
        convert_nums(diamond)
        pol2(diamond)

    d_list = numpy.array(d_list, dtype=float)
    ones_list = numpy.ones((len(d_list),1))
    d_list = numpy.hstack((ones_list,d_list))

    return d_list

def compute_params_normal_equation(diamonds):
    np_X = numpy.array(diamonds,dtype=float)
    #price = len(np_X[0])-1
    price = 10 
    np_Y = np_X[:,price]
    np_Y.transpose()
    np_X = numpy.delete(np_X, price, 1)
    np_Xt = np_X.transpose()

    np_XtX = numpy.dot(np_Xt, np_X)
    np_inv = numpy.linalg.inv(np_XtX)
    
    theta = numpy.dot(numpy.dot(np_inv, np_Xt), np_Y)
    return theta

def mean_squared_error(validation, theta):
    validation_X = numpy.array(validation,dtype=float)
    price = len(validation_X[0])-1
    price = 10 
    validation_Y = validation_X[:,price]
    validation_X = numpy.delete(validation_X, price, 1)

    theta_Y = numpy.dot(validation_X, theta)
    err = 0
    out = 0

    for i in range(0, len(theta_Y) - 1):
        err += (validation_Y[i] - theta_Y[i])*(validation_Y[i] - theta_Y[i])
        #if (err > 150000*150000):
        #    err -= (validation_Y[i] - theta_Y[i])*(validation_Y[i] - theta_Y[i])
        #    out += 1

    err = err/(2*(len(theta_Y) - out))

    print("Mean squared error = " + str(err))

def plot_results(validation, theta):
    validation_X = numpy.array(validation,dtype=float)
    price = len(validation_X[0])-1
    price = 10 
    validation_Y = validation_X[:,price]
    validation_X = numpy.delete(validation_X, price, 1)
    
    plt.scatter(validation_X[:,1], validation_Y, color='black')
    theta_predict_Y = numpy.dot(validation_X, theta)
    plt.scatter(validation_X[:,1], theta_predict_Y, color='blue')
    
    plt.xticks()
    plt.yticks()
    
    plt.show()


def main():
    if (len(sys.argv) != 3):
        print("ERROR: Usage python3 compute-diamonds.py (train-data) (test-data)")
        return

    diamonds = load_data(sys.argv[1])
    diamonds_train = diamonds[:len(diamonds)-8091]
    diamonds_valid = diamonds[len(diamonds)-8091:]
    validation = load_data(sys.argv[2])

    theta = compute_params_normal_equation(diamonds_train)

    print("Validation")
    mean_squared_error(diamonds_valid, theta)

    print("Test")
    mean_squared_error(validation, theta)
    plot_results(validation, theta)

    print(theta)

if __name__ == "__main__":
        main()
