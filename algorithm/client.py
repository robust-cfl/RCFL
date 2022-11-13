import torch
import csv
import sys
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from copy import deepcopy
from opacus import PrivacyEngine
from torch.autograd import Variable
from models.neural_nets import Digit5_MINE, Office_MINE, PACS_MINE, Home_MINE

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Client:
    def __init__(self, train_loader, test_loader, public_loader, model, hp, id_num):
        self.id = id_num
        # Belonging to which model.
        self.model_index = -1

        # Data_loaders
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.public_loader = public_loader

        # Model
        self.model = model
        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # minimizing mutual information between domain-invariant feature(embedding) and domain-specific feature embedding(embedding)
        if self.hp.dataset == 'digit5':
            self.mine = Digit5_MINE().cuda()
        elif self.hp.dataset == 'office':
            self.mine = Office_MINE().cuda()
        elif self.hp.dataset == 'home':
            self.mine = Home_MINE().cuda()
        else:
            assert self.hp.dataset == 'pacs'
            self.mine = PACS_MINE().cuda()

        self.training_loss = None
        self.training_acc = None

    def domain_adaptation(self, iterations, lr):
        model = self.model
        model.train()

        loss_class = torch.nn.NLLLoss().cuda()
        loss_domain = torch.nn.NLLLoss().cuda()

        # freeze domain-specific feature encoder theta_s
        for (key, param) in model.named_parameters():
            if key.startswith('specific'):
                param.requires_grad = False
        for (key, param) in self.mine.named_parameters():
            param.requires_grad = False

        optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, model.parameters())},
            {'params': filter(lambda p: p.requires_grad, self.mine.parameters())}
        ], lr=lr)

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

        for epoch in range(iterations):
            # assert len(self.public_train_loader) >= len(self.train_loader)
            len_dataloader = min(len(self.train_loader), len(self.public_loader))
            data_source_iter = iter(self.train_loader)
            data_target_iter = iter(self.public_loader)

            for i in range(len_dataloader):
                p = float(i + epoch * len_dataloader) / iterations / len_dataloader
                alpha = (2. / (1. + np.exp(-10 * p)) - 1)
                # training model using source data
                data_source = data_source_iter.next()
                s_img, s_label = data_source

                model.zero_grad()
                batch_size = len(s_label)

                domain_label = torch.zeros(batch_size).long()

                # move to gpu
                s_img, s_label, domain_label = s_img.cuda(), s_label.cuda(), domain_label.cuda()

                class_output, domain_output, _, _ = model(input_data=s_img, alpha=alpha)
                err_s_label = loss_class(class_output, s_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                # training model using target data
                data_target = data_target_iter.next()
                t_img, _ = data_target

                batch_size = len(t_img)

                domain_label = torch.ones(batch_size).long()

                # move to gpu
                t_img = t_img.cuda()
                domain_label = domain_label.cuda()

                _, domain_output, _, _ = model(input_data=t_img, alpha=alpha)
                err_t_domain = loss_domain(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()

                sys.stdout.write(
                    '\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                       err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
                sys.stdout.flush()

        # unfreeze domain-specific feature encoder theta_s
        for (key, param) in model.named_parameters():
            if key.startswith('specific'):
                param.requires_grad = True

    def classify_training(self, iterations, lr, warmup=False):
        model = self.model
        model.train()

        loss_class = torch.nn.NLLLoss().cuda()
        privacy_engine = None

        # freeze domain-invariant feature encoder theta_i and domain classifier theta_d
        if warmup:
            for (key, param) in model.named_parameters():
                if key.startswith('domain'):
                    param.requires_grad = False
            for (key, param) in self.mine.named_parameters():
                param.requires_grad = False
        else:
            for (key, param) in model.named_parameters():
                if key.startswith('invariant') or key.startswith('domain'):
                    param.requires_grad = False

        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, model.parameters())},
            {'params': filter(lambda p: p.requires_grad, self.mine.parameters())}
        ], lr=lr)

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

        for epoch in range(iterations):
            len_dataloader = len(tmp_train_loader)
            data_source_iter = iter(tmp_train_loader)

            for i in range(len_dataloader):
                # training model using source data
                data_source = data_source_iter.next()
                s_img, s_label = data_source

                optimizer.zero_grad()
                # move to gpu
                s_img, s_label = s_img.cuda(), s_label.cuda(),

                class_output, domain_output, domain_invariant_feature_embedding, domain_specific_feature_embedding = model(
                    input_data=s_img, alpha=0)
                # classification loss
                err_s_label = loss_class(class_output, s_label)
                # mutual information
                shuffle_domain_specific_feature_embedding = torch.index_select(domain_specific_feature_embedding, dim=0,
                                                                               index=Variable(torch.randperm(
                                                                                   domain_specific_feature_embedding.shape[
                                                                                       0])).cuda())
                mutual_information_loss = self.mutual_information_estimator(domain_invariant_feature_embedding,
                                                                            domain_specific_feature_embedding,
                                                                            shuffle_domain_specific_feature_embedding) / \
                                          s_label.shape[0]
                loss = err_s_label + self.hp.mi * mutual_information_loss
                loss.backward()
                optimizer.step()

                sys.stdout.write(
                    '\r epoch: %d, [iter: %d / all %d], err_s_label: %f' \
                    % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy()))
                sys.stdout.flush()

        # unfreeze domain-invariant feature encoder theta_i and domain classifier theta_d
        if warmup:
            for (key, param) in model.named_parameters():
                if key.startswith('domain'):
                    param.requires_grad = True
            for (key, param) in self.mine.named_parameters():
                param.requires_grad = False
        else:
            for (key, param) in model.named_parameters():
                if key.startswith('invariant') or key.startswith('domain'):
                    param.requires_grad = True
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
        # only update theta_i, theta_s, theta_y
        # tmp_params = deepcopy(self.model.state_dict())
        # for (key, value) in parameters:
        #     if key.startswith('domain'):
        #         continue
        #     else:
        #         tmp_params[key] = value
        # self.model.load_state_dict(tmp_params)
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

            class_output, _, _, _ = model(input_data=img, alpha=alpha)
            err = loss_class(class_output, label)
            loss_list.append(err.item())
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            n_total += batch_size

            i += 1
        acc = n_correct.data.numpy() * 1.0 / n_total
        return {'loss': sum(loss_list) / len(loss_list), 'accuracy': acc}

    def mutual_information_estimator(self, x, y, y_):
        joint, marginal = self.mine(x, y), self.mine(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))
