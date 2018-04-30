import sys
import os
import dataModelLib
import random
path = "../data/"
def get_train_test_data(train_data_points):
    x_train = []
    y_train = []
    x_train_m = []
    x_train_f = []
    y_train_m = []
    y_train_f = []
    # twitter data
    data = dataModelLib.load_data(path + "data.csv")
    data.remove(data[0])
    for i in data:
        p = i.split(",")
        if p[3] != '1':
            continue
        gen = p[2]
        if gen == "male":
            x_train_m.append(p[6])
            y_train_m.append(1)
        else:
            x_train_f.append(p[6])
            y_train_f.append(0)

    x_train_m = x_train_m\
            [0:min(len(x_train_m), train_data_points)]
    y_train_m = y_train_m\
            [0:min(len(y_train_m), train_data_points)]
    
    x_train_f = x_train_f\
            [0:min(len(x_train_f), train_data_points)]
    y_train_f = y_train_f\
            [0:min(len(y_train_f), train_data_points)]
    
    x_train = x_train_m + x_train_f
    y_train = y_train_m + y_train_f

    temp = []
    for i, w in enumerate(x_train):
        temp.append((x_train[i], y_train[i]))
    random.shuffle(temp)
    x_train = []
    y_train = []
    for i, w in enumerate(temp):
        x_train.append(w[0])
        y_train.append(w[1])

    data_1 = open(path + "mit.csv", 'r').read().split("\r\n")
    data_1.remove(data_1[0])
    data_2 = open(path + "stanford.csv", 'r').read().split("\r\n")
    data_2.remove(data_2[0])
    data = data_1 + data_2
    x_test = []
    y_test = []
    x_test_m = []
    y_test_m = []
    x_test_f = []
    y_test_f = []
    cnt = 0
    for i in data:
        p = i.split(",")
        p[1] = p[1].strip()
        if p[1] != 'm' and p[1] != 'f':
            continue
        if p[1] == 'm':
            x_test_m.append(p[2])
            y_test_m.append(0)
        else:
            x_test_f.append(p[2])
            y_test_f.append(1)
        x_test = x_test_f[0:300] + x_test_m[0:300]
        y_test = y_test_f[0:300] + y_test_m[0:300]
    #x_train, x_test, y_train, y_test = sklearn.cross_validation.train_test_split(x_test, y_test)
    return x_train, y_train, x_test, y_test


