import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# 逻辑回归
def LogisticRegression_train(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression(solver='liblinear')

    # 使用网格搜索找出更好的模型参数
    param_grid = [
        {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', 'l1'],
            'class_weight': ['balanced', None]
        }
    ]

    grid_search = GridSearchCV(log_reg, param_grid, cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # print(grid_search.best_estimator_)
    # print(grid_search.best_score_)
    # print(grid_search.best_params_)
    log_reg = grid_search.best_estimator_

    return log_reg, log_reg.score(X_test, y_test)


# KNN
def Knn_train(X_train, X_test, y_train, y_test):
    param_grid = [
        {
            'weights': ['uniform'],
            'n_neighbors': [i for i in range(1, 31)]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range(1, 31)],
            'p': [i for i in range(1, 6)]
        }
    ]
    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid)
    grid_search.fit(X_train, y_train)
    knn_clf = grid_search.best_estimator_
    return knn_clf, knn_clf.score(X_test, y_test)


# 决策树
def DecisionTreeClassifier_train(X_train, X_test, y_train, y_test):
    dt_clf = DecisionTreeClassifier(random_state=6)
    param_grid = [
        {
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        }
    ]
    grid_search = GridSearchCV(dt_clf, param_grid)

    grid_search.fit(X_train, y_train)
    dt_clf = grid_search.best_estimator_
    return dt_clf, dt_clf.score(X_test, y_test)


# 随机森林
def RandomForestClassifier_train(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier(oob_score=True, n_jobs=-1)
    param_grid = [
        {
            'n_estimators': range(1, 101, 10),
            'max_features': range(1, 21, 1)
        }
    ]
    grid_search = GridSearchCV(rf_clf, param_grid)

    grid_search.fit(X_train, y_train)
    rf_clf = grid_search.best_estimator_
    rf_clf.fit(X_train, y_train)
    return rf_clf, rf_clf.score(X_test, y_test)


# SVM
def svm_train(X_train, X_test, y_train, y_test):
    svc = svm.SVC(probability=True, gamma='auto')
    param_grid = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000]}
    ]
    grid_search = GridSearchCV(svc, param_grid)

    grid_search.fit(X_train, y_train)
    svc = grid_search.best_estimator_
    svc.fit(X_train, y_train)
    return svc, svc.score(X_test, y_test)


# 朴素贝叶斯
def GaussianNB_train(X_train, X_test, y_train, y_test):
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    return bayes, bayes.score(X_test, y_test)


# 计算F1
def F1Score(model, X_test, y_test):
    y_predict = model.predict(X_test)
    return f1_score(y_test, y_predict)


# 绘制混淆矩阵
def plot_cnf_matirx(model, X_test, y_test, description=''):
    y_predict = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)
    class_names = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create a heat map
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='OrRd',
                fmt='g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title(description, y=1.1, fontsize=16)
    plt.ylabel('实际值0/1', fontsize=12)
    plt.xlabel('预测值0/1', fontsize=12)
    plt.show()


if __name__ == '__main__':
    file = ['1.csv', '2.csv', '3.csv']
    result = []
    table = {}
    test_data = pd.read_csv('test.csv')
    X_test = test_data.drop(['target'], axis=1)
    y_test = test_data.target.values
    for index, path in enumerate(file):
        data = pd.read_csv(path)
        # 将特征与目标分开
        X_train = data.drop(['target'], axis=1)
        y_train = data.target.values
        # 分割数据集
        lg, lg_score = LogisticRegression_train(X_train, X_test, y_train, y_test)
        knn, knn_score = Knn_train(X_train, X_test, y_train, y_test)
        dt, dt_score = DecisionTreeClassifier_train(X_train, X_test, y_train, y_test)
        rf, rf_score = RandomForestClassifier_train(X_train, X_test, y_train, y_test)
        svc, svc_score = svm_train(X_train, X_test, y_train, y_test)
        bayse, bayse_score = GaussianNB_train(X_train, X_test, y_train, y_test)
        score = [lg_score, knn_score, dt_score, rf_score, svc_score, bayse_score]
        result.append(score)
        table['h' + str(index + 1)] = score
    df = pd.DataFrame(table, index=['lg', 'knn', 'dt', 'rf', 'svc', 'bayes'])
    print(df)
