import random

import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from load_image import load_grey_image


def compute_integral_map(image_list):
    integral_maps = []

    for image in image_list:
        inter_map = np.cumsum(np.cumsum(image, axis=0), axis=1, dtype=int)
        integral_maps.append(inter_map)
    return integral_maps


class Adaboost:
    def __init__(self, num_classifier, args):
        self.num_classifier = num_classifier
        self.models = []
        self.model_weight = []
        self.args = args

    def fit(self, image_features, labels):
        num_samples = len(image_features)
        sample_weight = np.ones(len(image_features)) / num_samples
        for i in range(self.num_classifier):
            with open(self.args.log_dir, "a") as f:
                print("第{}个弱分类器开始训练:".format(i + 1), file=f)
            weak_model = DecisionTreeClassifier(max_depth=1)
            weak_model.fit(image_features, labels, sample_weight=sample_weight)
            prediction = weak_model.predict(image_features)
            error = np.sum(sample_weight * (prediction != labels))
            if error == 0:
                error = 1e-10
            alpha = 0.5 * np.log((1 - error) / error)
            sample_weight *= np.exp(-alpha * labels * prediction)
            sample_weight /= np.sum(sample_weight)
            self.models.append(weak_model)
            self.model_weight.append(alpha)
        np.save(self.args.weight_dir, self.model_weight)
        np.save("models.npy", np.array(self.models))

    def __call__(self, images):
        prediction = np.zeros(images.shape[0])
        for weak_model, model_weight in zip(self.models, self.model_weight):
            prediction += model_weight * weak_model.predict(images)
        return prediction


class Haar_features:
    """
    仅包含basic和core的几个样式
    """

    def __init__(self, args):
        self.height = args.height
        self.width = args.width
        self.args = args
        self.kenel_sizes = args.kenel_sizes

    def extract_feature(self, image, method, kenel_size):
        if method == "two_vertical":
            feature_list = []
            for x in range(self.width - kenel_size):
                for y in range(self.height - kenel_size):
                    white = (
                        image[x + kenel_size // 2, y + kenel_size]
                        + image[x, y]
                        - image[x + kenel_size // 2, y]
                        - image[x, y + kenel_size]
                    )
                    black = (
                        image[x + kenel_size, y + kenel_size]
                        + image[x + kenel_size // 2, y]
                        - image[x + kenel_size, y]
                        - image[x + kenel_size // 2, y + kenel_size]
                    )

                    feature_list.append(white - black)

            return np.array(feature_list)
        elif method == "two_leveled":
            feature_list = []
            for x in range(self.width - kenel_size):
                for y in range(self.height - kenel_size):
                    white = (
                        image[x + kenel_size, y + kenel_size // 2]
                        + image[x, y]
                        - image[x + kenel_size, y]
                        - image[x, y + kenel_size // 2]
                    )
                    black = (
                        image[x + kenel_size, y + kenel_size]
                        + image[x, y + kenel_size // 2]
                        - image[x + kenel_size, y + kenel_size // 2]
                        - image[x, y + kenel_size]
                    )

                    feature_list.append(white - black)
            return np.array(feature_list)
        elif method == "three_vertical":
            feature_list = []
            for x in range(self.width - kenel_size):
                for y in range(self.height - kenel_size):
                    white = (
                        image[x + kenel_size // 3, y + kenel_size]
                        + image[x, y]
                        - image[x, y + kenel_size]
                        - image[x + kenel_size // 3, y]
                        + image[x + kenel_size, y + kenel_size]
                        + image[x + (kenel_size * 2) // 3, y]
                        - image[x + kenel_size, y]
                        - image[x + (kenel_size * 2) // 3, y + kenel_size]
                    )
                    black = (
                        image[x + (kenel_size * 2) // 3, y + kenel_size]
                        + image[x + kenel_size // 3, y]
                        - image[x + (kenel_size * 2) // 3, y]
                        - image[x + kenel_size // 3, y + kenel_size]
                    )

                    feature_list.append(white - 2 * black)
            return np.array(feature_list)
        elif method == "block":
            feature_list = []
            for x in range(self.width - kenel_size):
                for y in range(self.height - kenel_size):
                    white = (
                        image[x + kenel_size // 2, y + kenel_size // 2]
                        + image[x, y]
                        - image[x + kenel_size // 2, y]
                        - image[x, y + kenel_size // 2]
                        + image[x + kenel_size, y + kenel_size]
                        + image[x + kenel_size // 2, y + kenel_size // 2]
                        - image[x + kenel_size // 2, y + kenel_size]
                        - image[x + kenel_size, y + kenel_size // 2]
                    )
                    black = (
                        image[x + kenel_size, y + kenel_size // 2]
                        + image[x + kenel_size // 2, y]
                        - image[x + kenel_size, y]
                        - image[x + kenel_size // 2, y + kenel_size // 2]
                        + image[x + kenel_size // 2, y + kenel_size]
                        + image[x, y + kenel_size // 2]
                        - image[x + kenel_size // 2, y + kenel_size // 2]
                        - image[x, y + kenel_size]
                    )

                    feature_list.append(white - black)
            return np.array(feature_list)
        else:
            return ValueError("Unsupported Method")

    def __call__(self, image_list):
        integral_maps = compute_integral_map(image_list)
        image_features = []
        for image in integral_maps:
            features = []
            for kenel_size in self.kenel_sizes:
                TVfeatures = self.extract_feature(
                    image, method="two_vertical", kenel_size=kenel_size
                )
                TLfeatures = self.extract_feature(
                    image, method="two_leveled", kenel_size=kenel_size
                )
                BLfeatures = self.extract_feature(
                    image, method="block", kenel_size=kenel_size
                )
                ThVfeatures = self.extract_feature(
                    image, method="three_vertical", kenel_size=kenel_size
                )
                features = np.concatenate(
                    (features, TVfeatures, TLfeatures, BLfeatures, ThVfeatures)
                )
            features = np.array(features)
            features = features.flatten()
            image_features.append(features)
        image_features = np.array(image_features)
        return image_features


def haar_features(args):
    ratio = args.ratio

    image_list, image_labels = load_grey_image(args.img_root, args.not_face_dir)
    train_len = int(ratio * len(image_list))
    sample_seq = list(range(len(image_list)))
    random.shuffle(sample_seq)
    train_seq = sample_seq[0:train_len]
    val_seq = sample_seq[train_len:-1]
    image_list = [
        cv2.resize(image, (args.height, args.width), interpolation=3)
        for image in image_list
    ]

    train_list = [image_list[i] for i in train_seq]
    val_list = [image_list[i] for i in val_seq]

    train_label = [image_labels[i] for i in train_seq]
    val_label = [image_labels[i] for i in val_seq]
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    haar_extracter = Haar_features(args)

    train_features = haar_extracter(train_list)
    val_features = haar_extracter(val_list)
    with open(args.log_dir, "a") as file:
        print(
            "训练图片数:{},验证图片数:{}".format(
                len(train_features), len(val_features)
            ),
            file=file,
        )
    return train_features, val_features, train_label, val_label


def load_model(args):
    """
    加载训练完成的Adaboost模型
    """
    models_dir = args.models_dir
    models_weight_dir = args.weight_dir
    adaboost = Adaboost(args.num_classifier, args)
    models = np.load(models_dir, allow_pickle=True)
    weight = np.load(models_weight_dir)

    models = models.tolist()
    weight = weight.tolist()

    adaboost.models = models
    adaboost.model_weight = weight
    return adaboost


def draw_squares_imgs(args, img):
    """
    在不同的图像中识别人脸并画出方框
    输入格式：
    img = cv2.imread(dir)
    仅支持单张图片输入
    """
    height = img.shape[0]
    width = img.shape[1]
    img_copy = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sign_sizes = args.sign_sizes
    assert height > min(sign_sizes) and width > min(
        sign_sizes
    ), "图片太小，请使用分辨率较高的图片"
    haar_features = Haar_features(args)
    adaboost = load_model(args)
    records = []
    for sign_size in sign_sizes:
        for x in range(0, width - sign_size, 10):
            for y in range(0, height - sign_size, 10):
                patch = img[y : y + sign_size, x : x + sign_size]

                patch = cv2.resize(patch, (120, 120), interpolation=cv2.INTER_LINEAR)

                patch = patch.astype(int)
                patch = list([patch])
                features = haar_features(patch)
                prediction = adaboost(features)
                prediction = 1 / (1 + np.exp(-prediction))
                if prediction > 0.7:
                    records.append([x, y, sign_size, prediction])

        if len(records) != 0:
            break

    if not records:
        print("该图片没有人脸")
        return
    # for record in records:
    #     cv2.rectangle(
    #         img_copy,
    #         [record[0], record[1]],
    #         [record[0] + record[2], record[1] + record[2]],
    #         color=(0, 0, 255),
    #         thickness=2,
    #     )
    sorted_records = sorted(records, key=lambda x: x[3], reverse=True)
    top_two = sorted_records[:4]
    for record in top_two:
        cv2.rectangle(
            img_copy,
            [record[0], record[1]],
            [record[0] + record[2], record[1] + record[2]],
            color=(0, 0, 255),
            thickness=2,
        )
    cv2.imwrite("draw.jpg", img_copy)
