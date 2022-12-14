import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle


def feature_engineering(input_path, k, output_path=''):
    # 导入数据
    df = pd.read_csv(input_path)
    # 数据处理，对特征中非连续型数值进行处理
    first = pd.get_dummies(df['chest pain type'], prefix="cp")
    second = pd.get_dummies(df['ST slope'], prefix="slope")
    third = pd.get_dummies(df['resting ecg'], prefix="rc")
    df = pd.concat([df, first, second, third], axis=1)
    df = df.drop(columns=['chest pain type', 'ST slope', 'resting ecg'])

    # 将数据按行打乱
    df = shuffle(df).reset_index().drop(columns='index')

    n = len(df)-290 # 数据长度
    num = n // k

    # 获取数据集的特征
    columns = list(df.columns)
    columns.remove('target')

    # 数据标准化
    zscore = preprocessing.StandardScaler()
    df[columns] = zscore.fit_transform(df[columns])

    # 输出
    # for i in range(k):
    #     dt = df.iloc[i * num:(i + 1) * num]
    #     dt.to_csv(output_path + str(i + 1) + '.csv', encoding='utf-8', index=False)
    # df.iloc[k * num:].to_csv(output_path + 'test.csv', encoding='utf-8', index=False)
    df.iloc[:200].to_csv(output_path + '1.csv', encoding='utf-8', index=False)
    df.iloc[200:500].to_csv(output_path + '2.csv', encoding='utf-8', index=False)
    df.iloc[500 :900].to_csv(output_path + '3.csv', encoding='utf-8', index=False)
    df.iloc[900:].to_csv(output_path + 'test.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    input_path = 'heart_statlog_cleveland_hungary_final.csv'
    k = 3
    feature_engineering(input_path, k)
