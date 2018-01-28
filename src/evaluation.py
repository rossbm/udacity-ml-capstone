from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

#code adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

def roc_plot(y_true, y_preds, mdl_labels, out_path):
    n_models = len(mdl_labels)
    if n_models != len(y_preds):
        raise ValueError("Mismatch between number of provided labels and number of prediction sets.")

    #colors
    hsv = plt.get_cmap('hsv')
    colors = hsv(np.linspace(0, 1.0, n_models+1))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for ix, preds in enumerate(y_preds):
        fpr[ix], tpr[ix], _ = roc_curve(y_true, preds)
        roc_auc[ix] = auc(fpr[ix], tpr[ix])

    plt.figure()
    lw = 2
    for i in range(n_models):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                lw=2, label='ROC curve for {0} model (area = {1:0.4f})'.format(mdl_labels[i], roc_auc[i]))
    #45 degree line
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(out_path)
    return 