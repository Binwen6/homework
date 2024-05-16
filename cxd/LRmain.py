import argparse

from LR import train


def get_parser():
    parser = argparse.ArgumentParser(
        description="Face Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", default="/data/chenxd/MachineLearningHW/LR1", type=str
    )
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--loss_fn", default="BCELoss", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument(
        "--log_dir", default="/data/chenxd/MachineLearningHW/LR_train_log.txt", type=str
    )
    parser.add_argument(
        "--ckpt_dir",
        default="/data/chenxd/MachineLearningHW/LogisticRegression.pth",
        type=str,
    )
    parser.add_argument("--height", default=120, type=int)
    parser.add_argument("--width", default=120, type=int)
    parser.add_argument("--ratio", default=0.8, type=float)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--ROCdir", default="LR.jpg", type=str)

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    train(args)
