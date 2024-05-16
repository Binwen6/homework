# README

## 相关的库

scikit-learn, torch, torchvision, numpy等

## Haar&Adaboost分类

1. 使用方法

    ```
    python main.py --img_root your_img_root --not_face_dir your_img_dir...
    ```

    在控制终端输入以上命令即可运行，或修改main.py中`get_parser()`函数中的default值

    建议自己实现一个加载图片的程序

    如果采用本人所写的加载图片程序 请按照下列格式

    Positive Sample:

    ​	-- image0.jpg

    ​	-- image1.jpg

    ​	...

    Negative Sample:

    ​	-- class0

    ​		-- img0.jpg

    ​		-- img1.jpg

    ​		...

    ​	-- class1

    ​	....

2. 负样本来源：

​	建议采用Kaggle Natural Dataset [Natural Images | Kaggle](https://www.kaggle.com/datasets/prasunroy/natural-images) 下载完训练集后，将训练集中person的文件夹删除



## Logistic Regression 

1. 逻辑回归时的特征提取器采用了resnet18 运行前请确保电脑有独立的Nvidia显卡

2. 数据集准备（依旧推荐Kaggle）

    Dataset:

    ​	class0(人脸图像):

    ​		img0.jpg

    ​		img1.jpg

    ​		...

    ​	class1(非人脸图像):

    ​		img0.jpg

    ​		img1.jpg

​			...

3. 使用方法：

    ```
    python LRmain.py --(你所需要的参数调整)
    ```

    