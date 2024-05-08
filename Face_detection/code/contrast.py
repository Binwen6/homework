import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 数据集路径
dataset_path = 'D:\Code\Homework\data\Caltech_WebFaces'
annotations_file = os.path.join(dataset_path, 'D:\Code\Homework\data\WebFaces_GroundThruth.txt')

# 读取标注数据
def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            img_name = parts[0]
            faces = []
            for i in range(1, len(parts), 4):
                try:
                    x = int(float(parts[i]))
                    y = int(float(parts[i + 1]))
                    w = int(float(parts[i + 2]))
                    h = int(float(parts[i + 3]))
                    faces.append([x, y, w, h])
                except ValueError as e:
                    print(f"Skipping invalid annotation: {parts[i:i + 4]}")
            annotations.append((img_name, faces))
    return annotations

# 读取并标注图像
def load_and_annotate_images(dataset_path, annotations):
    images = []
    labels = []
    for img_name, faces in annotations:
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # 标注人脸区域
        for x, y, w, h in faces:
            if x + w <= img.shape[1] and y + h <= img.shape[0]:
                roi = cv2.resize(img[y:y + h, x:x + w], (24, 24))
                images.append(roi)
                labels.append(1)  # 人脸标签
        # 负样本区域（非人脸）
        num_neg_samples = len(faces) * 2
        for _ in range(num_neg_samples):
            x = np.random.randint(0, img.shape[1] - 24)
            y = np.random.randint(0, img.shape[0] - 24)
            roi = img[y:y + 24, x:x + 24]
            images.append(roi)
            labels.append(0)  # 非人脸标签
    return np.array(images), np.array(labels)

# Haar-like特征提取
def extract_haar_features(imgs):
    features = []
    for img in imgs:
        ii = cv2.integral(img)
        feat = []
        # 简单的Haar-like特征提取方式
        for size in [2, 4, 8, 16]:
            for i in range(0, 24 - size, size):
                for j in range(0, 24 - size, size):
                    s1 = ii[i, j] + ii[i + size, j + size] - ii[i, j + size] - ii[i + size, j]
                    feat.append(s1)
        features.append(feat)
    return np.array(features)

# 加载并提取数据
annotations = read_annotations(annotations_file)
images, labels = load_and_annotate_images(dataset_path, annotations)
features = extract_haar_features(images)

# 划分训练和测试集
split = int(0.8 * len(features))
X_train, y_train = features[:split], labels[:split]
X_test, y_test = features[split:], labels[split:]

# Adaboost分类器
def train_adaboost(X_train, y_train):
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=0.1
    )
    clf.fit(X_train, y_train)
    return clf

# Logistic回归
def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    return clf

# 计算ROC曲线
def compute_roc(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# 训练Adaboost和Logistic回归模型
adaboost_clf = train_adaboost(X_train, y_train)
logistic_clf = train_logistic_regression(X_train, y_train)

# 获取预测概率
y_pred_adaboost = adaboost_clf.predict_proba(X_test)[:, 1]
y_pred_logistic = logistic_clf.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_adaboost, tpr_adaboost, roc_auc_adaboost = compute_roc(y_test, y_pred_adaboost)
fpr_logistic, tpr_logistic, roc_auc_logistic = compute_roc(y_test, y_pred_logistic)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr_adaboost, tpr_adaboost, color='blue', lw=2, label=f'Adaboost (AUC = {roc_auc_adaboost:.2f})')
plt.plot(fpr_logistic, tpr_logistic, color='red', lw=2, label=f'Logistic Regression (AUC = {roc_auc_logistic:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()
