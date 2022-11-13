import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.utils import *
from copy import deepcopy
from data.data_utils import *
from sklearn import preprocessing
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Server:
    def __init__(self, model, hp, stats):
        # models
        self.model = model
        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.client_sizes = torch.Tensor(stats["split"]).cuda()

        self.updates = None

    def broadcast(self, clients):
        for client in clients:
            client.update(deepcopy(self.model.state_dict()))

    def train(self, clients, lr):
        updates = []
        for client in clients:
            print(client.id)
            theta = client.classify_training(lr=lr)
            updates += [(self.client_sizes[client.id], theta)]
        self.updates = updates

    def aggregate(self):
        updates = self.updates
        tmp_theta = deepcopy(self.model.state_dict())
        total_samples = 0

        for (size, theta) in updates:
            if total_samples == 0:
                # first client
                for k in tmp_theta.keys():
                    tmp_theta[k] = size * theta[k]
            else:
                # later client
                for k in tmp_theta.keys():
                    tmp_theta[k] += size * theta[k]

            total_samples += size

        for k in tmp_theta.keys():
            tmp_theta[k] /= total_samples

        # update server models
        self.model.load_state_dict(tmp_theta)

    def test(self, clients):
        training_loss_list, test_loss_list = [], []
        training_acc_list, test_acc_list = [], []
        for client in clients:
            results_train = client.validation(loader=client.train_loader)
            results_test = client.validation(loader=client.test_loader)
            training_loss_list += [results_train['loss']]
            training_acc_list += [results_train['accuracy']]

            test_loss_list += [results_test['loss']]
            test_acc_list += [results_test['accuracy']]
        clients_samples_num = [self.client_sizes[client.id] for client in clients]
        average_training_loss = sum(
            [clients_samples_num[i] * training_loss_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)
        average_training_acc = sum(
            [clients_samples_num[i] * training_acc_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)
        average_test_loss = sum(
            [clients_samples_num[i] * test_loss_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)
        average_test_acc = sum(
            [clients_samples_num[i] * test_acc_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)

        print(
            f"One-shot performance, training loss: {average_training_loss}, training acc: {average_training_acc}")
        print(f"One-shot performance, test loss: {average_test_loss}, test acc: {average_test_acc}")

        summary = {
            'training loss': average_training_loss,
            'training acc': average_training_acc,
            'test loss': average_test_loss,
            'test acc': average_test_acc
        }
        return summary

    def test_on_test_data(self, clients):
        test_loss_list, test_acc_list,  = [], []
        for client in clients:
            results_test = client.validation(loader=client.test_loader)
            test_loss_list += [results_test['loss']]
            test_acc_list += [results_test['accuracy']]
        clients_samples_num = [self.client_sizes[client.id] for client in clients]
        average_test_loss = sum(
            [clients_samples_num[i] * test_loss_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)
        average_test_acc = sum(
            [clients_samples_num[i] * test_acc_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)

        print(f"Standalone performance--test loss: {average_test_loss}, test acc: {average_test_acc}")

        summary = {
            'test loss': average_test_loss,
            'test acc': average_test_acc
        }
        return summary

    @staticmethod
    def determine_cluster_index_using_kmeans(X, n_cluster=4):
        # clustering in terms of cosine similarity, if l2 pls comment the following line
        estimator = KMeans(n_clusters=n_cluster)
        estimator.fit(X)
        labels_pred = estimator.labels_
        return labels_pred
