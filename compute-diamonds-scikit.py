import csv
import sys
import numpy
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import time

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

def load_data(input_diamonds):
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

    # Normalization
    for diamond in d_list:
        for i in range(0,9):
            diamond[i] = float(diamond[i])/max_params[i]

    return d_list

def compute_params_SGDR(diamonds, validation, _it):
    np_X = numpy.array(diamonds,dtype=float)
    np_X_validation = numpy.array(validation,dtype=float)

    np_Y = np_X[:,9]
    np_Y_validation = np_X_validation[:,9]

    np_Y.transpose()
    np_Y_validation.transpose()

    np_X = numpy.delete(np_X, 9, 1)
    np_X_validation = numpy.delete(np_X_validation, 9, 1)

    regr = linear_model.SGDRegressor(max_iter=_it, eta0=0.001)
    regr.fit(np_X, np_Y)
    diamonds_y_pred = regr.predict(np_X_validation)
    
    print('Coefficients: \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)

    print("Mean squared error: %.2f"
      % mean_squared_error(np_Y_validation, diamonds_y_pred))

    plt.scatter(np_X_validation[:,4], np_Y_validation, color='black')
    plt.plot(np_X_validation[:,4], diamonds_y_pred, color='blue', linewidth=3)

    plt.xticks()
    plt.yticks()

    #plt.show()

    return diamonds_y_pred 

def main():
    if (len(sys.argv) < 3):
        print("ERROR: Usage python3 compute-diamonds.py (train-data) (test-data) [iterations]")
        return

    diamonds = load_data(sys.argv[1])
    validation = load_data(sys.argv[2])
    if (len(sys.argv) == 3):
        iterations = 1000
    else:
        iterations = int(sys.argv[3])
        
    theta = compute_params_SGDR(diamonds, validation, iterations)

if __name__ == "__main__":
        main()
