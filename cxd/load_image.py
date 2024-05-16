import os

import cv2


def load_grey_image(img_root: str, not_face_dir: str):
    """
    img_root 为数据集根目录
    img_root:
        image001.jpg
        image002.jpg
        ...
    label_dir是存储标签的txt文件
    """
    Image_list = os.listdir(img_root)

    image_list = []
    for Image in Image_list:
        image_dir = os.path.join(img_root, Image)
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_list.append(image)
    no_face = []
    no_face_lists = os.listdir(not_face_dir)

    for no_face_list in no_face_lists:
        img_list = os.listdir(os.path.join(not_face_dir, no_face_list))
        for img in img_list:
            img_dir = os.path.join(not_face_dir, no_face_list, img)
            no_face.append(img_dir)
    nimages = []
    for i, img in enumerate(no_face):
        nimage = cv2.imread(img)
        nimage = cv2.cvtColor(nimage, cv2.COLOR_BGR2GRAY)
        nimages.append(nimage)
    label_list = []
    for _ in range(len(image_list)):
        label_list.append(1)
    for _ in range(len(nimages)):
        label_list.append(0)
    image_list.extend(nimages)
    return image_list, label_list
