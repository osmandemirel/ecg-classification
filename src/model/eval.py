import torch
import os
from pathlib import Path


def eval(model,dataloader, criteria, batch_size, gpu_f=True):
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
            loss = criteria(out, label)

            iter_preds = torch.where(out>0.05, 1, 0)

            preds = torch.cat((preds,iter_preds),dim=0)
            target = torch.cat((target,label),dim=0)
            eval_loss += float(loss.item())
            eval_correct_prediction += (iter_preds == label).sum().item()
        print("Validation: Loss:{} Accuracy:{}".format(eval_loss / len(dataloader)/batch_size,
                                                       eval_correct_prediction / len(dataloader)/batch_size))
    return (preds.cpu().numpy(),target.cpu().numpy())
