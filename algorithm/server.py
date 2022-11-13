import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils.utils import *
from copy import deepcopy
from data.data_utils import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Server:
    def __init__(self, public_loader, models, hp, stats):
        # public loader, unsupervised data
        self.public_loader = public_loader

        # models
        self.models = models
        # fedavg model
        self.avg_model = deepcopy(self.models[0])
        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.client_sizes = torch.Tensor(stats["split"]).cuda()

        self.ground_client_identities = stats["client_identities"]
        self.client_identities = [-1 for _ in range(len(self.client_sizes))]

        self.updates = None
        self.epsilons = None

    def broadcast(self, clients):
        for client in clients:
            # Broadcast model weights.
            cluster_index = self.client_identities[client.id]
            if cluster_index == -1:
                # using its initialized model (same across all the clients and the server)
                pass
            else:
                client.update(deepcopy(self.models[cluster_index].state_dict()))

    def fedavg_broadcast(self, clients):
        for client in clients:
            client.update(deepcopy(self.avg_model.state_dict()))

    def clustering(self, lambda_):
        updates = self.updates
        clustering_list = []
        for update in updates:
            if self.hp.isMode == 'add':
                client_vec = lambda_ * update['theta_i_vec'] + (1-lambda_) * update['theta_s_vec']
            else:
                assert self.hp.isMode == 'concat'
                client_vec = torch.cat((update['theta_i_vec'], update['theta_s_vec']))
            client_list = client_vec.tolist()
            clustering_list.append(client_list)
        points, client_identities = self.determine_cluster_index_using_kmeans(X=clustering_list, n_cluster=self.hp.K)
        ###
        self.client_identities = client_identities
        round_clustering_result = {k: [] for k in range(self.hp.K)}
        for c in range(len(client_identities)):
            cluster_index = client_identities[c]
            round_clustering_result[cluster_index] += [self.ground_client_identities[c]]
        purity = self.calculate_purity(round_clustering_result)
        print("Clustering Result", round_clustering_result, "Purity", purity)
        return points, client_identities, round_clustering_result, purity

    def train(self, clients, lr, fixClusters=False):
        updates = []
        for client in clients:
            if fixClusters:
                print(
                    f"client {client.id}, classifying for E1 + E2({self.hp.domainAdaptationEpoch + self.hp.classifyingEpoch}) epochs")
                client.classify_training(lr=lr, iterations=self.hp.domainAdaptationEpoch + self.hp.classifyingEpoch, warmup=True)
                print("")
            else:
                print(f"client {client.id}, domain adaptation for E1({self.hp.domainAdaptationEpoch}) epochs")
                client.domain_adaptation(lr=lr, iterations=self.hp.domainAdaptationEpoch)
                print("")
                print(f"client {client.id}, classifying for E2({self.hp.classifyingEpoch}) epochs")
                client.classify_training(lr=lr, iterations=self.hp.classifyingEpoch, warmup=False)
                print("")
            with torch.no_grad():
                theta_i_vec = torch.nn.utils.parameters_to_vector(client.model.invariant_feature.parameters())
                theta_s_vec = torch.nn.utils.parameters_to_vector(client.model.specific_feature.parameters())
            updates += [{
                'samples': self.client_sizes[client.id],
                'theta_i_vec': theta_i_vec,
                'theta_s_vec': theta_s_vec,
                'theta': client.model.state_dict()
            }]
        self.updates = updates

    def fedavg_train(self, clients, lr):
        updates = []
        epsilons = []
        for client in clients:
            print(f"\n{client.id} is warming up for E1 + E2({self.hp.domainAdaptationEpoch + self.hp.classifyingEpoch}) epochs\n")
            if self.hp.dp:
                theta, epsilon = client.classify_training(lr=lr, iterations=self.hp.domainAdaptationEpoch + self.hp.classifyingEpoch, warmup=True)
                epsilons.append(epsilon)
            else:
                theta = client.classify_training(lr=lr, iterations=self.hp.domainAdaptationEpoch + self.hp.classifyingEpoch, warmup=True)
            theta = deepcopy(theta)
            updates += [(self.client_sizes[client.id], theta)]
        self.updates = updates
        if self.hp.dp:
            self.epsilons = epsilons

    def aggregate(self):
        client_identities = self.client_identities
        updates = self.updates

        group_weight_list = [0 for _ in range(self.hp.K)]
        group_parameters_list = [deepcopy(self.models[i].state_dict()) for i in range(self.hp.K)]

        for j in range(len(updates)):
            identity = client_identities[j]
            size, theta = updates[j]['samples'], updates[j]['theta']
            if group_weight_list[identity] == 0:
                # first client
                for k in group_parameters_list[identity].keys():
                    group_parameters_list[identity][k] = size * theta[k]
            else:
                # later client
                for k in group_parameters_list[identity].keys():
                    group_parameters_list[identity][k] += size * theta[k]

            group_weight_list[identity] += size

        for k in range(len(group_parameters_list)):
            w = group_weight_list[k]
            parameters = group_parameters_list[k]
            for key in parameters.keys():
                parameters[key] /= w

        # update server models
        for j in range(len(self.models)):
            self.models[j].load_state_dict(group_parameters_list[j])

    def fedavg_aggregate(self):
        updates = self.updates
        tmp_theta = deepcopy(self.avg_model.state_dict())
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
        self.avg_model.load_state_dict(tmp_theta)

    def validation_on_target_domain(self, clients):
        loss_list, acc_list = [], []
        for client in clients:
            results = client.validation(loader=client.public_loader)
            loss_list += [results['loss']]
            acc_list += [results['accuracy']]

        clients_samples_num = [self.client_sizes[client.id] for client in clients]
        average_loss = sum(
            [clients_samples_num[i] * loss_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)
        average_acc = sum(
            [clients_samples_num[i] * acc_list[i] for i in range(len(clients_samples_num))]) / sum(
            clients_samples_num)

        print(f"Target loss: {average_loss}, Target acc: {average_acc}")

        summary = {
            'loss': average_loss,
            'acc': average_acc
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

        print(f"test loss: {average_test_loss}, test acc: {average_test_acc}")

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
            f"Training loss: {average_training_loss}, training acc: {average_training_acc}")
        print(f"Test loss: {average_test_loss}, test acc: {average_test_acc}")

        summary = {
            'training loss': average_training_loss,
            'training acc': average_training_acc,
            'test loss': average_test_loss,
            'test acc': average_test_acc
        }
        return summary

    @staticmethod
    def cosine_similarity(a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        if a.ndim == 1:
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
        elif a.ndim == 2:
            a_norm = np.linalg.norm(a, axis=1, keepdims=True)
            b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        else:
            raise RuntimeError("array dimensions {} not right".format(a.ndim))
        similarity = np.dot(a, b.T) / (a_norm * b_norm)
        return similarity

    @staticmethod
    def determine_cluster_index_using_kmeans(X, n_cluster=4, dim=3):
        # clustering in terms of cosine similarity, if l2 pls comment the following line
        pca = PCA(n_components=dim)
        X = pca.fit_transform(X)
        estimator = KMeans(n_clusters=n_cluster)
        estimator.fit(X)
        labels_pred = estimator.labels_
        return X.tolist(), labels_pred.tolist()

    @staticmethod
    def calculate_purity(round_clustering_result):
        # calculate purity
        # clusters_index
        c_set = set()
        for c_index, cluster in round_clustering_result.items():
            c_set.add(c_index)
        print(c_set)
        # round_clustering_result = {
        #     0: ['mnist', 'mnist', 'mnistm'],
        #     1: []
        # }
        # c_set = {'mnist', 'mnistm', 'svhn', 'usps', 'synthetic'}
        true, total = 0, 0
        for c_index, cluster in round_clustering_result.items():
            cluster_info = {}
            for member in cluster:
                if member in cluster_info:
                    cluster_info[member] = cluster_info[member] + 1
                else:
                    cluster_info[member] = 1
            values = cluster_info.values()
            total += sum(values)
            true += max(values)
        return true / total
