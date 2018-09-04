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

    max_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for diamond in d_list:
        convert_nums(diamond)
        for i in range(0,9):
            if float(diamond[i]) > max_params[i] and i != 9:
                max_params[i] = float(diamond[i])
                
    # Doesn't normalize price
    max_params[9] = 1
                
    d_list = numpy.array(d_list, dtype=float)
    
    #Normalization
    for diamond in d_list:
        for i in range(0,9):
            diamond[i] = float((diamond[i])-max_params[i]/2)/max_params[i]

    output_prices.append(d_list[:,9])
    d_list = numpy.delete(d_list, 9, 1)

    ones_list = numpy.ones((len(d_list),1))
    d_list = numpy.hstack((ones_list,d_list))

    return d_list

def compute_params_SGDR(diamonds, prices, validation, validation_prices, _it):
    np_X = numpy.array(diamonds,dtype=float)
    np_X_validation = numpy.array(validation,dtype=float)

    np_Y = prices
    np_Y_validation = validation_prices 

    np_Y.transpose()
    np_Y_validation.transpose()

    regr = linear_model.SGDRegressor(max_iter=_it, eta0=0.001, verbose=1)
    regr.fit(np_X, np_Y)
    diamonds_y_pred = regr.predict(np_X_validation)
    
    print('Coefficients: \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)

    print("Mean squared error: %.2f"
      % mean_squared_error(np_Y_validation, diamonds_y_pred))
    print("R2 Score: %.2f"
      % r2_score(np_Y_validation, diamonds_y_pred))

    #plt.scatter(np_X_validation[:,1], np_Y_validation, color='black')
    #plt.scatter(np_X_validation[:,1], diamonds_y_pred, color='blue')

    #plt.xticks()
    #plt.yticks()

    #plt.show()

    return regr 

def _mean_squared_error(validation, prices, regr):
    validation_X = numpy.array(validation,dtype=float)
    validation_Y = prices 
    print(validation_Y)

    theta_Y = regr.predict(validation_X)

    print("Mean squared error: %.2f"
      % mean_squared_error(validation_Y, theta_Y))
    print("R2 Score: %.2f"
      % r2_score(validation_Y, theta_Y))

def print_stars():
    stars = ['*'] * 80
    for star in stars:
        sys.stdout.write(star)
    sys.stdout.flush()
    print()

def main():
    if (len(sys.argv) < 3):
        print("ERROR: Usage python3 compute-diamonds.py (train-data) (test-data) [iterations]")
        return

    mask = 1023
    if (len(sys.argv) == 3):
        iterations = 1
    else:
        iterations = int(sys.argv[3])
        
    print_stars()
    print("Training and Validation")
    diamonds_prices = []
    diamonds = load_data(sys.argv[1], diamonds_prices, mask)
    diamonds_prices = numpy.array(diamonds_prices[0], dtype=float)

    diamonds_train = diamonds[:len(diamonds)-8091]
    diamonds_valid = diamonds[len(diamonds)-8091:]
    diamonds_prices_train = diamonds_prices[:len(diamonds_prices)-8091]
    diamonds_prices_valid = diamonds_prices[len(diamonds_prices)-8091:]

    regr = compute_params_SGDR(diamonds_train, diamonds_prices_train, diamonds_valid, diamonds_prices_valid, iterations)

    print_stars()
    print("Testing")
    validation_prices = []
    validation = load_data(sys.argv[2], validation_prices, mask)
    validation_prices = numpy.array(validation_prices[0], dtype=float)
    _mean_squared_error(validation, validation_prices, regr)

if __name__ == "__main__":
        main()
