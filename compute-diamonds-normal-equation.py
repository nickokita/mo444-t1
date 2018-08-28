import csv
import sys
import numpy
import matplotlib.pyplot as plt
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

def compute_params_normal_equation(diamonds):
    np_X = numpy.array(diamonds,dtype=float)
    np_Y = np_X[:,9]
    np_Y.transpose()
    np_X = numpy.delete(np_X, 9, 1)
    np_Xt = np_X.transpose()

    np_XtX = numpy.dot(np_Xt, np_X)
    np_inv = numpy.linalg.inv(np_XtX)
    
    theta = numpy.dot(numpy.dot(np_inv, np_Xt), np_Y)
    return theta

def mean_squared_error(validation, theta):
    validation_X = numpy.array(validation,dtype=float)
    validation_Y = validation_X[:,9]
    validation_X = numpy.delete(validation_X, 9, 1)

    theta_Y = numpy.dot(validation_X, theta)
    err = 0

    for i in range(0, len(theta_Y) - 1):
        err += (validation_Y[i] - theta_Y[i])*(validation_Y[i] - theta_Y[i])

    err = err/len(theta_Y)

    print("Mean squared error = " + str(err))

def plot_results(validation, theta):
    validation_X = numpy.array(validation,dtype=float)
    validation_Y = validation_X[:,9]
    validation_X = numpy.delete(validation_X, 9, 1)
    
    plt.scatter(validation_X[:,4], validation_Y, color='black')
    theta_predict_Y = numpy.dot(validation_X, theta)
    plt.plot(validation_X[:,4], theta_predict_Y, color='blue', linewidth=3)
    
    plt.xticks()
    plt.yticks()
    
    plt.show()


def main():
    if (len(sys.argv) != 3):
        print("ERROR: Usage python3 compute-diamonds.py (train-data) (test-data)")
        return

    diamonds = load_data(sys.argv[1])
    validation = load_data(sys.argv[2])
    theta = compute_params_normal_equation(diamonds)

    mean_squared_error(validation, theta)
    plot_results(validation, theta)

    print(theta)

if __name__ == "__main__":
        main()
