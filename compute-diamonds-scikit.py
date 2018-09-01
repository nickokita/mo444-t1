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
    _size = len(diamond)-2
    for i in range(0,_size):
        for j in range(i,_size):
            diamond.append(float(diamond[i])*float(diamond[j]))
            for k in range(k,_size):
                diamond.append(float(diamond[i])*float(diamond[j])*float(diamond[k]))
    return diamond

def load_data(input_diamonds):
    f_diamonds = open(input_diamonds, newline='')
    d_reader = csv.reader(f_diamonds, delimiter=',')
    d_list = list(d_reader)
    d_list.pop(0)

    max_params = [0] * 100
    for diamond in d_list:
        convert_nums(diamond)
        pol3(diamond)
        for i in range(0,len(diamond)):
            if float(diamond[i]) > float(max_params[i]) and i != 9:
                max_params[i] = diamond[i]

    max_params[9] = 1
    for diamond in d_list:
        for i in range(0, len(diamond)):
            diamond[i] = float(diamond[i])/float(max_params[i])

    d_list = numpy.array(d_list, dtype=float)
    ones_list = numpy.ones((len(d_list),1))
    d_list = numpy.hstack((ones_list,d_list))

    return d_list

def compute_params_SGDR(diamonds, validation, _it):
    price = 10
    np_X = numpy.array(diamonds,dtype=float)
    np_X_validation = numpy.array(validation,dtype=float)

    np_Y = np_X[:,price]
    np_Y_validation = np_X_validation[:,price]

    np_Y.transpose()
    np_Y_validation.transpose()

    np_X = numpy.delete(np_X, price, 1)
    np_X_validation = numpy.delete(np_X_validation, price, 1)

    regr = linear_model.SGDRegressor(max_iter=_it, eta0=0.001)
    regr.fit(np_X, np_Y)
    diamonds_y_pred = regr.predict(np_X_validation)
    
    print('Coefficients: \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)

    print("Mean squared error: %.2f"
      % mean_squared_error(np_Y_validation, diamonds_y_pred))

    plt.scatter(np_X_validation[:,1], np_Y_validation, color='black')
    plt.scatter(np_X_validation[:,1], diamonds_y_pred, color='blue')

    plt.xticks()
    plt.yticks()

    theta = numpy.hstack((regr.intercept_, regr.coef_))

    #plt.show()

    return theta 

def _mean_squared_error(validation, theta):
    validation_X = numpy.array(validation,dtype=float)
    price = 10 
    validation_Y = validation_X[:,price]
    validation_X = numpy.delete(validation_X, price, 1)

    theta_Y = numpy.dot(validation_X, theta)
    err = 0
    out = 0

    for i in range(0, len(theta_Y) - 1):
        err += (validation_Y[i] - theta_Y[i])*(validation_Y[i] - theta_Y[i])
        if (err > 150000*150000):
            err -= (validation_Y[i] - theta_Y[i])*(validation_Y[i] - theta_Y[i])
            out += 1

    err = err/(2*(len(theta_Y) - out))

    print("Mean squared error = " + str(err))

def main():
    if (len(sys.argv) < 3):
        print("ERROR: Usage python3 compute-diamonds.py (train-data) (test-data) [iterations]")
        return

    diamonds = load_data(sys.argv[1])
    diamonds_train = diamonds[:len(diamonds)-8091]
    diamonds_valid = diamonds[len(diamonds)-8091:]
    if (len(sys.argv) == 3):
        iterations = 1000
    else:
        iterations = int(sys.argv[3])
        
    theta = compute_params_SGDR(diamonds_train, diamonds_valid, iterations)

    validation = load_data(sys.argv[2])
    print("Test")
    _mean_squared_error(validation, theta)

if __name__ == "__main__":
        main()
