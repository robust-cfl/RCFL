import torch
import csv
import sys
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from copy import deepcopy
from opacus import PrivacyEngine
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Client:
    def __init__(self, train_loader, test_loader, model, hp, id_num):
        self.id = id_num

        # Data_loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Model
        self.model = model

        # fedavg domain-classifier用不到，塞到optimizer parameters中会导致这部分参数没有梯度，进而导致dp没法用，所以把这部分参数去掉

        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.w_glob_keys = self.set_w_glob_keys()

    def set_w_glob_keys(self):
        keys = list(self.model.state_dict().keys())
        keys = list(reversed(keys))  # [top -> down(near the data)]
        tmp_keys = []
        for k in keys:
            if k.startswith('invariant_feature'):
                tmp_keys.append(k)
        print(tmp_keys)
        return tmp_keys

    def classify_training(self, lr):
        model = self.model
        model.train()
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        privacy_engine = None

        loss_class = torch.nn.NLLLoss().cuda()
        iterations = self.hp.classifyingEpoch

        if self.hp.dp:
            privacy_engine = PrivacyEngine()
            model, optimizer, tmp_train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=self.hp.sigma,
                max_grad_norm=self.hp.max_per_sample_grad_norm,
            )

        else:
            tmp_train_loader = self.train_loader

        head_eps = int(iterations // 2)
        for epoch in range(iterations):
            # training head first
            if epoch < head_eps:
                for name, param in model.named_parameters():
                    if name in self.w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            # then training base
            else:
                for name, param in model.named_parameters():
                    if name in self.w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            len_dataloader = len(tmp_train_loader)
            data_source_iter = iter(tmp_train_loader)
            for i in range(len_dataloader):
                optimizer.zero_grad()
                # training model using source data
                data_source = data_source_iter.next()
                s_img, s_label = data_source
                # move to gpu
                s_img, s_label = s_img.cuda(), s_label.cuda()

                class_output = model(input_data=s_img)
                err_s_label = loss_class(class_output, s_label)
                err_s_label.backward()
                optimizer.step()

                sys.stdout.write(
                    '\r epoch: %d, [iter: %d / all %d], err_s_label: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy()))
                sys.stdout.flush()
        if self.hp.dp:
            epsilon, best_alpha = privacy_engine.accountant.get_privacy_spent(
                delta=self.hp.delta
            )
            print(
                # f"Train Epoch: {epoch} \t"
                # f"Loss: {np.mean(losses):.6f} "
                f"(ε = {epsilon:.2f}, δ = {self.hp.delta}) for α = {best_alpha}"
            )

            # summary = {
            #     # 'Train/Loss': np.mean(losses),
            #     'ε': epsilon,
            #     'δ': self.hp.delta,
            #     'α': best_alpha
            # }
            # print(summary)
            return deepcopy(model.state_dict()), epsilon
        else:
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

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def set_shared_params(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key in self.w_glob_keys:
                tmp_params[key] = value
        self.set_params(tmp_params)