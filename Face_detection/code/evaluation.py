from face_detection import evaluate_image
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(test_image, model, scale, stepSize, windowSize, true_face_boxes):
    tpr_list = []
    fpr_list = []

    for threshold in np.linspace(0, 1, num=10):
        tpr, fpr = evaluate_image(test_image, model, scale, stepSize, windowSize, true_face_boxes, threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    plt.figure()
    plt.plot(fpr_list, tpr_list, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
