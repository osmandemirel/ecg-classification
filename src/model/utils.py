import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import numpy as np

def save_confusion_matrix(confisuon_matrix,title,filename):
    # plot confusion matrix
    fig,ax = plt.subplots(figsize = (10,7))
    ax.matshow(confisuon_matrix, cmap=plt.cm.Blues)
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            c = confisuon_matrix[i][j]
            ax.text(j, i, str(c), va='center', ha='center')
            time.sleep(0.5)
    fig.savefig(filename+'.png')

def calculate_confusion_matrix(preds,target):
    """
    :param preds: prediction array
    :param label: target array
    :return: confusion matrix
    """
    assert preds.shape==target.shape
    return confusion_matrix(target,preds)

def calculate_accuracy(preds,target):
    """
    :param preds: prediction array
    :param label: target array
    :return: accuracy
    """
    assert preds.shape==target.shape
    return (preds==target).sum()/preds.shape[0]

def calculate_f1_measure(preds,target,labels=None):
    """
    :param preds: prediction array
    :param label: target array
    :return: accuracy
    """
    assert preds.shape == target.shape
    return f1_score(y_true=target,y_pred=preds,labels=labels,average='macro')

def calculate_recall(preds,target,labels=None):
    """
    :param preds: prediction array
    :param label: target array
    :return: recall
    """
    assert preds.shape == target.shape
    return recall_score(y_true=target,y_pred=preds,labels=labels,average='macro')

def calculate_precision(preds,target,labels=None):
    """
    :param preds: prediction array
    :param label: target array
    :return: precision
    """
    assert preds.shape == target.shape
    return precision_score(y_true=target,y_pred=preds,labels=labels,average='macro')
"""
def test_scores():
    preds = np.array([3,0,1,1,2,3,0,0,1,2])
    targets = np.array([3,0,1,1,2,3,0,0,1,2])

    precision = calculate_precision(preds,targets)
    recall = calculate_recall(preds,targets)
    accuracy = calculate_accuracy(preds,targets)
    conf_mat = calculate_confusion_matrix(preds,targets)
    assert precision == 1
    assert recall == 1
    assert accuracy == 1
    assert conf_mat == np.array([[3,0,0,0],[0,3,0,0],[0,0,2,0],[0,0,0,2]])

test_scores()
"""
