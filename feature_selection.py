# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 上午2:56
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN


def main():
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94], class_sep=0.8, random_state=0)
    ros = RandomOverSampler(random_state=0)

    y = y.reshape(-1, 1)
    print(X.shape)
    print(y.shape)

    X_resampled, y_resampled = ros.fit_sample(X, y)

    print(X.shape)

    X_resampled, y_resampled = SMOTE().fit_sample(X, y)
    print(X_resampled.shape)
    print(y_resampled.shape)


if __name__ == '__main__':
    main()
