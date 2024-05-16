import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def draw_ROC_curve(args, prediction, val_labels):
    # ROCCurve = []
    # thresholds = np.linspace(-1, max(prediction), 100)

    # for threshold in thresholds:
    #     output = (prediction > threshold).astype(int)
    #     output = output.reshape(-1)
    #     val_labels = val_labels.reshape(-1)
    #     true_positive = np.sum((output == 1) & (val_labels == 1))

    #     false_positive = np.sum((output == 1) & (val_labels == 0))
    #     true_negative = np.sum((output == 0) & (val_labels == 0))
    #     false_negative = np.sum((output == 0) & (val_labels == 1))

    #     true_positive_ratio = true_positive / (true_positive + false_negative)
    #     false_positive_ratio = false_positive / (true_negative + false_positive)
    #     ROCCurve.append([true_positive_ratio, false_positive_ratio])

    # ROCCurve = np.array(ROCCurve)

    fpr, tpr, threshold = roc_curve(val_labels, prediction)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="AUC:{}".format(roc_auc))
    plt.legend()
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(args.ROCdir)
