import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_processing import drawing_face_box, select_negative_samples
from feature_extraction import resolve_haar_feature
from face_detection import pyramid, sliding_window, compute_iou, evaluate_image
from evaluation import plot_roc_curve


image_folder = 'D:\Code\Homework\data\Caltech_WebFaces'
labels_file = 'D:\Code\Homework\data\WebFaces_GroundThruth.txt'

def main(folder, file):
    face_box = drawing_face_box(file)
    negative_samples = select_negative_samples(folder, face_box, num_samples=2)

    features = []
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        feature_vector = [
            resolve_haar_feature(image, feature_type, face_box[image_name])
            for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']
        ]
        features.append(np.array(feature_vector).flatten())

    negative_features = []
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        bounding_box_1, bounding_box_2 = negative_samples[image_name]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        feature_vector_1 = [
            resolve_haar_feature(image, feature_type, bounding_box_1)
            for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']
        ]
        feature_vector_2 = [
            resolve_haar_feature(image, feature_type, bounding_box_2)
            for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']
        ]
        negative_features.append(np.array(feature_vector_1).flatten())
        negative_features.append(np.array(feature_vector_2).flatten())

    # Combine data and labels
    X = np.array(features + negative_features)
    y = np.array([1] * len(features) + [-1] * len(negative_features))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Define the Adaboost model's pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Feature scaling
        ('adaboost', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50))  # Adaboost classifier
    ])

    # Fit the pipeline on the training set
    pipeline.fit(X_train, y_train)



    # 设置参数
    scale = 1.5
    stepSize = 10
    windowSize = (100, 100)
    iou_threshold = 0.5

    # 主函数
    true_positives_total = 0
    face_boxes_total = 0
    false_positives_total = 0
    positives_total = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 读取图像
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            # 使用 basename 只提取文件名
            filename_only = os.path.basename(filename)
            face_boxes = [face_box.get(filename_only, None)]
            face_boxes = [box for box in face_boxes if box is not None]
                
            # 计算 true_positives, face_boxes, false_positives 和 positives
            true_positives, false_positives, positives, face_boxes = evaluate_image(image, pipeline, scale, stepSize, windowSize, face_boxes, iou_threshold)

            # 将当前图像的结果添加到总计中
            true_positives_total += true_positives
            false_positives_total += false_positives
            positives_total.extend(positives)
            face_boxes_total += len(face_boxes)

    # 计算整体的 FPR 和 TPR
    overall_TPR = true_positives_total / face_boxes_total
    overall_FPR = false_positives_total / len(positives_total)

    print("Overall TPR:", overall_TPR)
    print("Overall FPR:", overall_FPR)


    # 绘制 RoC 曲线
    plt.plot(overall_FPR, overall_TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (RoC) Curve')
    plt.show()


if __name__ == "__main__":
    main(image_folder, labels_file)
