import numpy as np
import cv2

def resolve_haar_feature(image, feature_type, coodinates):
    (x, y, width, height) = map(int, coodinates)  # 获取窗口的坐标和尺寸
    # image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # print(x, y , width, height)
    if feature_type == 'vertical_edge':  # 如果特征类型是'vertical_edge'
        # Divide the window into two rectangles:  # 将窗口分成两个矩形
        left = image[y:y+height, x:x+width//2]  # 左半部分矩形
        right = image[y:y+height, x+width//2:x+width]  # 右半部分矩形
        # Calculate the difference in pixel sums between the two halves  # 计算两个矩形之间像素和的差值
        feature_value = np.sum(right, dtype=np.int64) - np.sum(left, dtype=np.int64)  # 计算特征值
        return feature_value  # 返回特征值
    if feature_type == 'horizontal_edge':  # 如果特征类型是'horizontal_edge'
        # Divide the window into three rectangles:  # 将窗口分成两个矩形
        bottom = image[y:y+height//2, x:x+width]  # 下部分矩形
        top = image[y+height//2:y+height, x:x+width]  # 上部分矩形
        # Calculate the difference in pixel sums between the top and bottom thirds  # 计算上部分和下部分的像素和之差
        feature_value = np.sum(top, dtype=np.int64) - np.sum(bottom, dtype=np.int64)# 计算特征值
        return feature_value  # 返回特征值
    if feature_type == 'diagonal_edge':  # 如果特征类型是'diagonal_edge'
        # Divide the window into four rectangles:  # 将窗口分成四个矩形
        top_left = image[y:y+height//2, x:x+width//2]  # 左上部分矩形
        top_right = image[y:y+height//2, x+width//2:x+width]  # 右上部分矩形
        bottom_left = image[y+height//2:y+height, x:x+width//2]  # 左下部分矩形
        bottom_right = image[y+height//2:y+height, x+width//2:x+width]  # 右下部分矩形
        # Calculate the difference in pixel sums between the center and the surrounding rectangles  # 计算中心矩形和周围矩形之间像素和的差值
        feature_value = np.sum(top_left, dtype=np.int64) + np.sum(bottom_right, dtype=np.int64) - np.sum(top_right, dtype=np.int64) - np.sum(bottom_left, dtype=np.int64)  # 计算特征值
        return feature_value  # 返回特征值
    if feature_type == 'center_surround':  # 如果特征类型是'center_surround'
        center = image[y+height//4:y+3*height//4, x+width//4:x+3*width//4]  # 中心部分矩形
        surround = image[y:y+height, x:x+width]  # 周围部分矩形
        # Calculate the difference in pixel sums between the center and the surrounding rectangles  # 计算中心矩形和周围矩形之间像素和的差值
        feature_value = np.sum(center, dtype=np.int64) - np.sum(surround, dtype=np.int64)  # 计算特征值
        return feature_value  # 返回特征值
    else:
        raise ValueError("Unsupported feature type")  # 如果特征类型不支持，则引发ValueError异常