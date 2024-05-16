import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

from drawROCcurve import draw_ROC_curve

device = torch.device(0)


def hook(module, input, output):
    features = output
    return features


class ModifiedResnet(nn.Module):
    def __init__(self):
        super(ModifiedResnet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)

    def forward(self, image):
        return torch.sigmoid(self.resnet(image))


def get_images(args):
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize([args.height, args.width], interpolation=3),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFolder(dataset_dir, transform=transform)

    train_len = int(len(dataset) * args.ratio)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def get_loss_fn():
    return torch.nn.BCELoss()


def get_optimizer(params, lr):
    return torch.optim.SGD(params, lr)


def test_model(args, val_loader, model):
    model = model.cuda()
    model.eval()
    outputs = []
    val_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = model(imgs)
            outputs.append(output)
            val_labels.append(labels)

    val_labels = torch.cat(val_labels)
    outputs = torch.cat(outputs)
    val_labels = val_labels.cpu()
    outputs = outputs.cpu()
    val_labels = torch.Tensor.numpy(val_labels)
    outputs = torch.Tensor.numpy(outputs)

    draw_ROC_curve(args, outputs, val_labels)


def train(args):
    train_loader, val_loader = get_images(args)
    loss_fn = get_loss_fn()
    model = ModifiedResnet()
    model.train()
    optimizer = get_optimizer(model.parameters(), 0.01)
    with open(args.log_dir, "a") as f:
        print(datetime.datetime.now(), file=f)
        print("特征提取器:{}".format(args.model), file=f)
        print("损失函数:{}".format(args.loss_fn), file=f)
        print("设备:GPU{}".format(args.device), file=f)

    for epoch in range(args.epoch):
        with open(args.log_dir, "a") as f:
            print("第{}轮训练开始:".format(epoch + 1), file=f)
        model = model.to(device)
        total_train_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            output = model(imgs)
            loss = loss_fn(output, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss
        with open(args.log_dir, "a") as f:
            print("训练集损失:{}".format(total_train_loss), file=f)
        with torch.no_grad():
            model.eval()
            model = model.to(device)
            total_val_loss = 0
            correct = 0
            total = 0
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1)
                output = model(imgs)

                loss = loss_fn(output, labels.float())
                total_val_loss += loss
                prediction = output > 0.5
                correct += (prediction == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total
            with open(args.log_dir, "a") as f:
                print(
                    "验证集损失:{},正确率:{}".format(total_val_loss, accuracy), file=f
                )
                print("-----------------------------------------------------", file=f)
    torch.save(model.state_dict(), args.ckpt_dir)
    print("训练完成,模型保存路径为:{}".format(args.ckpt_dir))
    test_model(args, val_loader, model)
