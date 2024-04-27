from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import numpy as np


def train_model(features, negative_features):
    
    # Combine data and labels
    X = np.array(features + negative_features)
    y = np.array([1] * len(features) + [-1] * len(negative_features))


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 定义Adaboost模型的pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 特征缩放
        ('adaboost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50))  # Adaboost分类器
    ])

    # 在训练集上拟合pipeline
    pipeline.fit(X_train, y_train)
    


