import torch
import csv
import sys
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Client:
    def __init__(self, train_loader, test_loader, model, hp, id_num):
        self.id = id_num
        # Belonging to which model.
        self.model_index = -1

        # Data_loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Model
        self.model = model
        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def model_difference(start_point, new_point):
        loss = 0
        old_params = start_point.state_dict()
        for name, param in new_point.named_parameters():
            loss += torch.norm(old_params[name] - param, 2)
        return loss

    def classify_training(self, lr):
        model = self.model
        model.train()
        startPoint = deepcopy(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_class = torch.nn.NLLLoss().cuda()
        iterations = self.hp['classifyingEpoch']

        for epoch in range(iterations):
            len_dataloader = len(self.train_loader)
            data_source_iter = iter(self.train_loader)

            for i in range(len_dataloader):
                # training model using source data
                data_source = data_source_iter.next()
                s_img, s_label = data_source

                model.zero_grad()
                # move to gpu
                s_img, s_label = s_img.cuda(), s_label.cuda(),

                class_output = model(input_data=s_img)
                err_s_label = loss_class(class_output, s_label)
                difference = self.model_difference(startPoint, model)
                total_loss = err_s_label + 0.1 * difference
                total_loss.backward()
                optimizer.step()

                sys.stdout.write(
                    '\r epoch: %d, [iter: %d / all %d], err_s_label: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy()))
                sys.stdout.flush()
        return deepcopy(model.state_dict())

    def update(self, parameters):
        self.model.load_state_dict(parameters)

    def validation(self, loader=None):
        loss_class = torch.nn.NLLLoss().cuda()
        loss_list = []
        if loader is None:
            loader = self.test_loader
        alpha = 0
        model = self.model
        model.eval()

        len_dataloader = len(loader)
        data_iter = iter(loader)

        i = 0
        n_total = 0
        n_correct = 0

        while i < len_dataloader:
            # test model using source/target data
            data = data_iter.next()
            img, label = data

            batch_size = len(label)

            img = img.cuda()
            label = label.cuda()

            class_output = model(input_data=img, alpha=alpha)
            err = loss_class(class_output, label)
            loss_list.append(err.item())
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            n_total += batch_size

            i += 1
        acc = n_correct.data.numpy() * 1.0 / n_total
        return {'loss': sum(loss_list) / len(loss_list), 'accuracy': acc}
