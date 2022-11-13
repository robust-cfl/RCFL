import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.utils import *
from copy import deepcopy
from data.data_utils import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from torch.nn.utils import parameters_to_vector
import models.neural_nets as neural_nets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Server:
    def __init__(self, hp, stats):
        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.client_sizes = torch.Tensor(stats["split"]).cuda()

        self.num_clusters = hp.K
        self.centers = self.setup_centers()

        self.updates = None
        self.epsilons = None

    def setup_centers(self):
        centers = []
        for i in range(self.num_clusters):
            # affect server initialization
            setup_seed(self.hp.seed + i)
            centers.append(getattr(neural_nets, self.hp.model)().to(device))
        return centers

    def update_centers(self, clients):
        cluster_params_lst = {i: [] for i in range(self.num_clusters)}
        for client in clients:
            dis = np.array([])
            for c in self.centers:
                dis = np.append(dis, torch.norm(
                    parameters_to_vector(deepcopy(client.model).parameters())
                    - parameters_to_vector(deepcopy(c).parameters()),
                    p=2, dim=0).data.cpu().numpy())
            client.model_index = np.argmin(dis)
            cluster_params_lst[client.model_index].append((self.client_sizes[client.id], client.get_params()))
        for client in clients:
            print(client.model_index)
        for i in range(self.num_clusters):
            if len(cluster_params_lst[i]) != 0:
                updated_center_params = self.aggregate(cluster_params_lst[i])
                self.centers[i].load_state_dict(updated_center_params)

    def broadcast(self, clients):
        for client in clients:
            client.update(deepcopy(self.centers[client.model_index].state_dict()))

    def train(self, clients, lr):
        updates = []
        epsilons = []
        for client in clients:
            print(client.id)
            if self.hp.dp:
                theta, epsilon = client.classify_training(lr=lr)
                epsilons.append(epsilon)
            else:
                theta = client.classify_training(lr=lr)
            updates += [(self.client_sizes[client.id], theta)]
        self.updates = updates
        if self.hp.dp:
            self.epsilons = epsilons

    def aggregate(self, updates):
        tmp_theta = deepcopy(updates[0][1])
        total_samples = 0

        for (size, theta) in updates:
            if total_samples == 0:
                # first client
                for k in tmp_theta.keys():
                    if self.hp.dp:
                        t_k = '_module.' + k
                        tmp_theta[k] = size * theta[t_k]  # 去掉——module.
                    else:
                        tmp_theta[k] = size * theta[k]
            else:
                # later client
                for k in tmp_theta.keys():
                    if self.hp.dp:
                        t_k = '_module.' + k
                        tmp_theta[k] += size * theta[t_k]
                    else:
                        tmp_theta[k] += size * theta[k]

            total_samples += size

        for k in tmp_theta.keys():
            tmp_theta[k] /= total_samples

        # update server models
        return tmp_theta

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

        print(f"Performance--test loss: {average_test_loss}, test acc: {average_test_acc}")

        summary = {
            'test loss': average_test_loss,
            'test acc': average_test_acc
        }
        return summary

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

    @staticmethod
    def determine_cluster_index_using_kmeans(X, n_cluster=4):
        estimator = KMeans(n_clusters=n_cluster)
        estimator.fit(X)
        labels_pred = estimator.labels_
        return labels_pred
