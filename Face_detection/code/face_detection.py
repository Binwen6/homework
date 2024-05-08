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
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - boxA: tuple/list of (x1, y1, x2, y2) coordinates for the first box
    - boxB: tuple/list of (x1, y1, x2, y2) coordinates for the second box

    Returns:
    - iou: float value representing the IoU between boxA and boxB
    """

    # Determine the (x, y) coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of the intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



import numpy as np

def evaluate_image(image, model, scale, stepSize, windowSize, face_boxes, iou_threshold=0.5):
    positives = []

    # 生成图像金字塔并滑动窗口进行检测
    for resized in pyramid(image, scale=scale):
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=windowSize):
            if window.shape[0] != windowSize[1] or window.shape[1] != windowSize[0]:
                continue

            features = []
            for feature_type in ['vertical_edge', 'horizontal_edge', 'diagonal_edge', 'center_surround']:
                feature = resolve_haar_feature(window, feature_type, (x, y, window.shape[0], window.shape[1]))
                features.append(feature)

            # 将特征数组展平并调整为匹配模型输入的形状
            features_array = np.array(features).reshape(1, -1)
            pred = model.predict(features_array)

            if pred == 1:
                positives.append((x, y, x + windowSize[0], y + windowSize[1]))


    true_positives = 0
    false_positives = 0
    detected = []

    for pred_box in positives:
        for true_box in face_boxes or []:
            iou = compute_iou(pred_box, true_box)
            if iou > iou_threshold:
                if true_box not in detected:
                    detected.append(true_box)
                    true_positives += 1
                    break
        else:
            false_positives += 1

    return true_positives, false_positives, positives, face_boxes



