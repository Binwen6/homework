import cv2
import numpy as np
from feature_extraction import resolve_haar_feature
import matplotlib.pyplot as plt
from feature_extraction import resolve_haar_feature




def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, (w, w))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate_image(image, model, scale, stepSize, windowSize, face_boxes, iou_threshold=0.5):
    positives = []
    for resized in pyramid(image, scale=scale):
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=windowSize):
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue
            # 假设您的特征提取和模型预测代码如下
            features = resolve_haar_feature(window)
            pred = model.predict([features])
            if pred == 1:
                positives.append((x, y, x + windowSize[0], y + windowSize[1]))

    true_positives = 0
    false_positives = 0
    detected = []

    for pred_box in positives:
        for true_box in face_boxes:
            iou = compute_iou(pred_box, true_box)
            if iou > iou_threshold:
                if true_box not in detected:
                    detected.append(true_box)
                    true_positives += 1
                    break
        else:
            false_positives += 1

    if len(positives) == 0:
        return 0, 0  # 避免除以零的错误
    TPR = true_positives / len(face_boxes)
    FPR = false_positives / len(positives)
    return TPR, FPR



