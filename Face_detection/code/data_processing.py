import re
import random
import cv2
import os



# 对labels_file中的每组坐标进行计算，算出每行image对应的人脸范围，存储在face_box中
def drawing_face_box(labels_file):
    # 读取并解析标签文件
    face_box = {}
    with open(labels_file, 'r') as file:  # 打开标签文件
        for line in file:  # 逐行读取文件内容
            match = re.match(r'(pic\d+\.jpg)\s+([\d\.e\+\s]+)', line)  # 使用正则表达式匹配行中的图片文件名和标签
            if match:  # 如果匹配成功
                image_name = match.group(1)  # 获取图片文件名
                labels = match.group(2).strip().split()  # 获取标签，并去除首尾空格，然后按空格分割为列表
                labels = [float(label) for label in labels]  # 将标签列表中的每个元素转换为浮点数
                if len(labels) == 8:  # 如果标签列表长度为8，即包含了所有坐标
                    x1, y1, x2, y2, x3, y3, x4, y4 = labels  # 将标签列表中的元素分别赋值给对应的变量
                    x_min = min(x1, x2, x3, x4)  # 计算x坐标的最小值
                    x_max = max(x1, x2, x3, x4)  # 计算x坐标的最大值
                    y_min = min(y1, y2, y3, y4)  # 计算y坐标的最小值
                    y_max = max(y1, y2, y3, y4)  # 计算y坐标的最大值
                    face_box[image_name] = (x_min, y_min, x_max - x_min, y_max - y_min)  # 将图片文件名和对应的标签存入字典
    return face_box


# 从dataset_path中依次读取image，并按照image文件名在labels_file中按行进行匹
## 2.5 负样本构造
def select_negative_samples(image_folder, face_box, num_samples=2):
    samples = {}  # Initialize an empty list to store negative samples
    for image_name in os.listdir(image_folder):  # Iterate over each image in the folder
        # if image_name not in face_box:  # Check if the image does not have a face bounding box
        image_path = os.path.join(image_folder, image_name)  # Get the full path of the image
        image = cv2.imread(image_path)  # Read the image using OpenCV
        if image is None:  # Check if the image is not empty
            continue  # Skip to the next image if the current image is empty
        img_height, img_width = image.shape[:2]  # Get the height and width of the image
        sample_w = 10  # Generate a random width for the negative sample
        sample_h = 10  # Generate a random height for the negative sample
        sample_x1 = 0
        sample_y1 = 0
        sample_x2 = img_width - sample_w
        sample_y2 = img_height - sample_h
        samples[image_name] = ((sample_x1, sample_y1, sample_w, sample_h), (sample_x2, sample_y2, sample_w, sample_h))
    return samples  # Return the list of negative samples