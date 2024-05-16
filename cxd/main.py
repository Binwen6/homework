import argparse
import datetime

import cv2
import numpy as np

from drawROCcurve import draw_ROC_curve
from Haar_Adaboost import Adaboost, draw_squares_imgs, haar_features


def get_parser():
    parser = argparse.ArgumentParser(
        description="Face Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img_root",
        default="/data/chenxd/MachineLearningHW/Caltech_WebFaces",
        type=str,
        help="正样本的路径",
    )
    parser.add_argument(
        "--not_face_dir",
        default="/data/chenxd/MachineLearningHW/natural_images",
        type=str,
        helo="负样本的路径",
    )
    parser.add_argument(
        "--height", default=120, type=int, help="图像的高将被调整大小到height来进行训练"
    )
    parser.add_argument(
        "--width", default=120, type=int, help="图像的宽将被调整到width来进行训练"
    )

    parser.add_argument(
        "--kenel_sizes", default=[15, 30, 60, 120], help="Haar算子的大小"
    )
    parser.add_argument(
        "--ratio", default=0.8, type=float, help="划分训练集和验证集的比例"
    )

    parser.add_argument(
        "--num_classifier", default=100, type=int, help="Adaboost中弱分类器的数目"
    )
    parser.add_argument(
        "--log_dir",
        default="/data/chenxd/MachineLearningHW/train_log.txt",
        type=str,
        help="训练路径的保存地址，尾部记得加.txt",
    )
    parser.add_argument(
        "--ROCdir", default="AdaROC.jpg", type=str, help="ROC曲线保存地址"
    )
    parser.add_argument(
        "--weight_dir", default="weight.npy", type=str, help="模型权重保存地址"
    )
    parser.add_argument(
        "--sign_sizes", default=[180], type=list[int], help="画图时滑动窗口的大小"
    )
    parser.add_argument("--models_dir", default="models.npy", help="模型保存地址")

    return parser


def train():
    args = get_parser().parse_args()
    with open(args.log_dir, "a") as file:
        print(datetime.datetime.now(), file=file)
        print("基本信息:", file=file)
        print("Adaboost 弱分类器数量：{}".format(args.num_classifier), file=file)
        print("提取特征开始", file=file)
    train_features, val_features, train_labels, val_labels = haar_features(args)
    with open(args.log_dir, "a") as file:
        print("提取特征结束", file=file)
        print("开始训练:", file=file)
    model = Adaboost(args.num_classifier, args)
    model.fit(train_features, train_labels)

    prediction = model(val_features)
    draw_ROC_curve(args, prediction, val_labels)
    prediction = prediction.reshape(-1)
    prediction = prediction > 0.5

    accuracy = np.sum(prediction == val_labels) / len(val_labels)

    with open(args.log_dir, "a") as file:
        print("准确度:{}".format(accuracy))
        print("权重文件保存为:{}".format(args.weight_dir))


if __name__ == "__main__":
    train()
    args = get_parser().parse_args()

    img = cv2.imread("/data/chenxd/MachineLearningHW/draw_squares/pic00423.jpg")
    draw_squares_imgs(args, img)
