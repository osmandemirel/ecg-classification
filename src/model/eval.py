import torch
from src.model.utils import *
import os
from pathlib import Path


def eval(model,dataloader, criteria, batch_size,gpu_f=True):
    """
    :param model: model to be tested
    :param dataloader: dataloader for testing set
    :param criteria: loss function to be used
    :param batch_size: batchsize in dataloader
    :return: return predictions and targets as tuple
    """
    eval_loss = float(0.0)
    eval_correct_prediction = 0

    if torch.cuda.is_available() and gpu_f:
        gpu_flag = True
    else:
        gpu_flag = False

    if gpu_flag:
        model.to("cuda:0")

    model.eval()
    with torch.no_grad():
        # save for confusion matrix
        if gpu_flag:
            preds = torch.tensor([]).to(torch.device('cuda:0'))
            target = torch.tensor([]).to(torch.device('cuda:0'))
        else:
            preds = torch.tensor([])
            target = torch.tensor([])

        for data, label in dataloader:
            if gpu_flag:
                data = data.to(torch.device('cuda:0'))
                label = label.to(torch.device('cuda:0'))

            out = model(data)

            if str(criteria) == "CrossEntropyLoss()":
                loss = criteria(out, label)
            else:
                loss = criteria(out.float(), label.float().view(-1, 1))
            # save metrics
            iter_preds = torch.max(out.detach(), 1).indices

            preds = torch.cat((preds,iter_preds),dim=0)
            target = torch.cat((target,label),dim=0)
            eval_loss += float(loss.item())
            eval_correct_prediction += (iter_preds == label).int().sum().item()
        print("Validation: Loss:{} Accuracy:{}".format(eval_loss / len(dataloader)/batch_size,
                                                       eval_correct_prediction / len(dataloader)/batch_size))
    return (preds.cpu().numpy(),target.cpu().numpy())

def eval_scores(preds,targets,path=None,file_name=None):
    """
    :param preds: predictions, should be a numpy array
    :param targets: targets, should be a numpy array
    :param path: file path to save metrics
    :param file_name: file name to save metrics
    :return: return scores as tuple
        (accuracy,precision,recall,confusion matrix)
    """
    assert preds.shape==targets.shape
    precision = calculate_precision(preds,targets)
    recall = calculate_recall(preds,targets)
    accuracy = calculate_accuracy(preds,targets)
    conf_mat = calculate_confusion_matrix(preds,targets)

    if isinstance(path,str) and isinstance(file_name,str): # save if path and filename are provided
        with open(os.path.join(path, file_name)+".txt", "w") as file:
            file.write(str(accuracy) + "\n")
            file.write(str(precision) + "\n")
            file.write(str(recall) + "\n")
            file.write(str(conf_mat) + "\n")
            file.close()
            print("Results are written to {}.txt".format(file_name))
        save_confusion_matrix(conf_mat,"Confusion Matrix",os.path.join(path, file_name))
    return (accuracy,precision,recall,conf_mat)
