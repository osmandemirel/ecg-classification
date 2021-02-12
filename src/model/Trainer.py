import os
import torch
from torch import optim
from pathlib import Path
from src.data.dataset import ChestXrayDataset


class Trainer:
    def __init__(self, model, criteria, train_loader, val_loader, config):
        """
        :param model: A deep learning model extends to nn.Module
        :param criteria: A loss function
        :param train_loader: Data loader for training data set
        :param val_loader: Data loader for validation data set
        :param config: A dict data that includes configurations. You can refer to below:
            conf = {
                "number_of_epochs": 50,
                "val_every": 5,
                "optimizer": "SGD",
                "lr": 0.004,
                "device": "gpu",
                "result_path":"path_string",
                "batch_size":16
            }
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criteria = criteria
        self.config = config
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        self.gpu_flag = self.__set_device()
        if self.gpu_flag:
            self.model.to(torch.device('cuda:0'))
        self.optimizer = self.__set_optimizer()
        self.RESULT_SAVE_PATH = self.config["result_path"]

    def train_supervised(self,n_epochs=None):
        if n_epochs is None:
            n_epochs = self.config["number_of_epochs"]
        val_every = self.config["val_every"] if "val_every" in self.config else 5

        for epoch in range(n_epochs):
            self.train_epoch()
            print("Epoch:{} Loss:{} Accuracy:{}".format(epoch, self.metrics["train_loss"][epoch],
                                                        self.metrics["train_acc"][epoch]))
            if (epoch+1) % val_every == 0:
                self.optimizer.zero_grad(set_to_none=True)
                self.evaluate()
                self.save_status()

    def train_epoch(self):
        iter_loss = float(0.0)
        iter_correct_prediction = int(0)
        self.model.train()
        
        for data, label in self.train_loader:
            
            self.optimizer.zero_grad()
            self.model.zero_grad()
            if self.gpu_flag:
                data = data.to(torch.device('cuda:0'))
                label = label.to(torch.device('cuda:0'))

            out = self.model(data)

            if str(self.criteria)=="CrossEntropyLoss()":
                loss = self.criteria(out, label)
            else:
                loss = self.criteria(out.float(), label.float().view(-1, 1))

            loss.backward()
            self.optimizer.step()

            iter_loss += float(loss.item())
            iter_correct_prediction += (torch.max(out.detach(), 1).indices==label).int().sum().item()
        
        self.metrics["train_loss"].append(iter_loss / len(self.train_loader))    
        self.metrics["train_acc"].append(iter_correct_prediction / (len(self.train_loader) * self.config["batch_size"]))
        torch.cuda.empty_cache()

    def evaluate(self):
        val_loss = float(0.0)
        val_correct_prediction = 0
        self.model.eval()
        with torch.no_grad():
            for data, label in self.val_loader:
                if self.gpu_flag:
                    data = data.to(torch.device('cuda:0'))
                    label = label.to(torch.device('cuda:0'))

                out = self.model(data)

                if str(self.criteria) == "CrossEntropyLoss()":
                    loss = self.criteria(out, label)
                else:
                    loss = self.criteria(out.float(), label.float().view(-1, 1))

                val_loss += float(loss.item())
                val_correct_prediction += (torch.max(out.detach(), 1).indices==label).int().sum().item()
            print("Validation: Loss:{} Accuracy:{}".format(val_loss / len(self.val_loader),
                                                        val_correct_prediction / len(self.val_loader) / self.config["batch_size"]))

            self.metrics["val_loss"].append(val_loss / len(self.val_loader))
            self.metrics["val_acc"].append(val_correct_prediction / (len(self.val_loader) * self.config["batch_size"]))

    def __set_optimizer(self):
        weight_decay = self.config["weight_decay"] if "weight_decay" in self.config else 0
        if self.config["optimizer"]=="SGD":
            momentum = self.config["momentum"] if "momentum" in self.config else 0
            optimizer = optim.SGD(
                params=self.model.parameters(),lr=self.config["lr"],
                weight_decay=weight_decay,momentum=momentum
            )
        else:
            optimizer = optim.Adam(
                params=self.model.parameters(),
                lr = self.config["lr"],
                weight_decay=weight_decay
            )
        return optimizer

    def __set_device(self):
        config_device = self.config["device"] if "device" in self.config else "cpu"
        if torch.cuda.is_available() and config_device=="gpu":
            gpu_flag=True
        else:
            gpu_flag=False
        return gpu_flag

    def save_status(self):
        torch.save(self.model.state_dict(), os.path.join(self.RESULT_SAVE_PATH, "model.pth"))
        self.save_metrics()

    def save_metrics(self):
        with open( os.path.join(self.RESULT_SAVE_PATH, "metrics.txt"), "w") as file:
            file.write(str(self.metrics)+"\n")
            file.close()

