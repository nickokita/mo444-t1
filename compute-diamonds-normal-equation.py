import csv
import sys
import numpy
#import matplotlib.pyplot as plt
import time

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
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
    _size = len(diamond)-1
    for i in range(0,_size):
        diamond.append(float(diamond[i])*float(diamond[i]))
    return diamond

def select_features(d_list, mask):
    features = []
    for i in range(len(d_list[0])-1,-1,-1):
        select = 2**i
        if mask & select:
            features.append(i)
        else:
            d_list = numpy.delete(d_list, i, 1)

    print("Features selected = " + str(features))
    return d_list

def load_data(input_diamonds, output_prices, mask):
    f_diamonds = open(input_diamonds, newline='')
    d_reader = csv.reader(f_diamonds, delimiter=',')
    d_list = list(d_reader)
    d_list.pop(0)


    for diamond in d_list:
        convert_nums(diamond)

    d_list = numpy.array(d_list, dtype=float)
    output_prices.append(d_list[:,9])
    d_list = numpy.delete(d_list, 9, 1)

    d_list = select_features(d_list, mask)

    ones_list = numpy.ones((len(d_list),1))
    d_list = numpy.hstack((ones_list,d_list))

    return d_list

def compute_params_normal_equation(diamonds, prices):
    np_X = numpy.array(diamonds,dtype=float)
    np_Y = prices
    np_Y.transpose()
    np_Xt = np_X.transpose()

    np_XtX = numpy.dot(np_Xt, np_X)
    np_inv = numpy.linalg.inv(np_XtX)
    
    theta = numpy.dot(numpy.dot(np_inv, np_Xt), np_Y)
    return theta

def _mean_squared_error(validation, prices, theta):
    validation_X = numpy.array(validation,dtype=float)
    validation_Y = prices 

    theta_Y = numpy.dot(validation_X, theta)

    print("Mean squared error: %.2f"
      % mean_squared_error(validation_Y, theta_Y))
    print("R2 Score: %.2f"
      % r2_score(validation_Y, theta_Y))

def plot_results(validation, prices, theta):
    validation_X = numpy.array(validation,dtype=float)
    validation_Y = prices
    
    #plt.scatter(validation_X[:,1], validation_Y, color='black')
    theta_predict_Y = numpy.dot(validation_X, theta)
    #plt.scatter(validation_X[:,1], theta_predict_Y, color='blue')
    
    #plt.xticks()
    #plt.yticks()
    
    #plt.show()

def print_stars():
    stars = ['*'] * 80
    for star in stars:
        sys.stdout.write(star)
    sys.stdout.flush()
    print()

def main():
    if (len(sys.argv) != 4):
        print("ERROR: Usage python3 compute-diamonds.py (train-data) (test-data) mask")
        return

    print_stars()
    mask = int(sys.argv[3])
    print("Mask = " + str(mask))
    print_stars()
    print("Training and Validation")
    diamonds_prices = []
    diamonds = load_data(sys.argv[1], diamonds_prices, mask)
    diamonds_prices = numpy.array(diamonds_prices[0], dtype=float)

    diamonds_train = diamonds[:len(diamonds)-8091]
    diamonds_valid = diamonds[len(diamonds)-8091:]
    diamonds_prices_train = diamonds_prices[:len(diamonds_prices)-8091]
    diamonds_prices_valid = diamonds_prices[len(diamonds_prices)-8091:]

    theta = compute_params_normal_equation(diamonds_train, diamonds_prices_train)

    _mean_squared_error(diamonds_valid, diamonds_prices_valid, theta)

    print_stars()
    print("Testing")
    validation_prices = []
    validation = load_data(sys.argv[2], validation_prices, mask)
    validation_prices = numpy.array(validation_prices[0], dtype=float)
    _mean_squared_error(validation, validation_prices, theta)
    plot_results(validation, validation_prices, theta)

    print_stars()
    print("Model: " + str(theta))
    print_stars()

if __name__ == "__main__":
        main()
