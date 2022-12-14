import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def feature_zuhe(X):
    n, m = X.shape
    mat = np.ones((n, 3, m, m))
    for k in range(X.shape[0]):
        x = X[k]
        for i in range(m):
            for j in range(m):
                mat[k][0][i][j] = x[i]
                mat[k][1][i][j] = x[j]
                mat[k][2][i][j] = x[i] * x[j]
    return mat


def load_data1():
    # 路径
    train_path = r'/Users/otto/Desktop/研究生申请材料汇总/FederatedLearning/1.csv'
    test_path = r'/Users/otto/Desktop/研究生申请材料汇总/FederatedLearning/test.csv'
    # 读取文件
    train_data = pd.read_csv(train_path)
    X_train = np.array(train_data.drop(['target'], axis=1))
    y_train = np.array(train_data.target.values)
    test_data = pd.read_csv(test_path)
    X_test = np.array(test_data.drop(['target'], axis=1))
    y_test = np.array(test_data.target.values)
    # 特征组合
    X_train = feature_zuhe(X_train)
    X_test = feature_zuhe(X_test)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    # 把从csv文件读取的数据封装成(feature, label)的dataset
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset)
    return train_loader, test_loader


def load_data2():
    # 路径
    train_path = r'/Users/otto/Desktop/研究生申请材料汇总/FederatedLearning/2.csv'
    test_path = r'/Users/otto/Desktop/研究生申请材料汇总/FederatedLearning/test.csv'
    # 读取文件
    train_data = pd.read_csv(train_path)
    X_train = np.array(train_data.drop(['target'], axis=1))
    y_train = np.array(train_data.target.values)
    test_data = pd.read_csv(test_path)
    X_test = np.array(test_data.drop(['target'], axis=1))
    y_test = np.array(test_data.target.values)
    # 特征组合
    X_train = feature_zuhe(X_train)
    X_test = feature_zuhe(X_test)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    # 把从csv文件读取的数据封装成(feature, label)的dataset
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset)
    return train_loader, test_loader


def load_data3():
    # 路径
    train_path = r'/Users/otto/Desktop/研究生申请材料汇总/FederatedLearning/3.csv'
    test_path = r'/Users/otto/Desktop/研究生申请材料汇总/FederatedLearning/test.csv'
    # 读取文件
    train_data = pd.read_csv(train_path)
    X_train = np.array(train_data.drop(['target'], axis=1))
    y_train = np.array(train_data.target.values)
    test_data = pd.read_csv(test_path)
    X_test = np.array(test_data.drop(['target'], axis=1))
    y_test = np.array(test_data.target.values)
    # 特征组合
    X_train = feature_zuhe(X_train)
    X_test = feature_zuhe(X_test)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    # 把从csv文件读取的数据封装成(feature, label)的dataset
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset)
    return train_loader, test_loader
