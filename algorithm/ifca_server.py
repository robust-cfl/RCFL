import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from data.data_utils import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pickle
from tqdm import tqdm
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Server:
    def __init__(self, models, hp, stats):
        # models
        self.models = models
        # Hyper-parameters
        self.hp = hp
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.client_sizes = torch.Tensor(stats["split"]).cuda()

        # IFCA中不需要用到
        self.client_identities = [-1 for _ in range(len(self.client_sizes))]

        self.updates = None

        self.cluster_info = dict()

    def broadcast(self, clients, c_round, stats):
        self.cluster_info[f"{c_round}"] = dict()

        pbar = tqdm(clients, ncols=100)
        for client in pbar:
            pbar.set_description(f"Client#{client.id} is choosing cluster model")
            # determine_server_model for ifca (choose the smallest loss)
            client.determine_server_model(self.models)  # 将k个模型发送给client，然后选出loss最小的模型以及对应的id

        # ==================== cluster info ========================
        for i in range(0, self.hp.K):  # cluster: 1~k个
            self.cluster_info[f"{c_round}"][f"#{i}#"] = []

        for client in clients:
            self.cluster_info[f"{c_round}"][f"#{client.cluster_id}#"].append(
                (client.id, stats['users_cluster_identity'][client.id]))
        print("Current round cluster info: ", self.cluster_info[f"{c_round}"])
        print("Each cluster's client number:")
        for key, value in self.cluster_info[f"{c_round}"].items():
            print(f"{key}: {len(value)}")

        # 每轮保存下最新的cluster_info到data目录下（多次run，会覆盖obj）
        cluster_info_path = "../data/cluster_info.obj"
        with open(cluster_info_path, 'wb') as fp:
            pickle.dump(self.cluster_info, fp)

        # 保存到wandb run文件夹下
        wandb_file_path = os.path.join(wandb.run.dir, "cluster_info_dict.obj")
        with open(wandb_file_path, 'wb') as fp:
            pickle.dump(self.cluster_info, fp)
        # 同步到云端
        wandb.save("cluster_info_dict.obj")
        # ============================================

    def mixed_clustering(self, clients):
        # train_acc_list = []
        # test_acc_list = []
        for client in clients:
            client.get_identity_vec_mixed()
            # train_acc_list += [train_acc]
            # test_acc_list += [test_acc]
        # print(f"average train acc: {sum(train_acc_list) / len(train_acc_list)},"
        #       f"average test acc: {sum(test_acc_list) / len(test_acc_list)}")
        clustering_list = [client.identity_vec.tolist() for client in clients]
        cluster_identities = self.determine_cluster_index_using_kmeans(X=clustering_list, n_cluster=self.hp['K'])
        clients_ids = [client.id for client in clients]

        # record client's (have seen) cluster identity
        for i in range(len(clients_ids)):
            client_id = clients_ids[i]
            identity = cluster_identities[i]
            self.client_identities[client_id] = identity

    def specific_clustering(self, clients):
        # train_acc_list = []
        # test_acc_list = []
        for client in clients:
            client.get_identity_vec_specific()
            # train_acc_list += [train_acc]
            # test_acc_list += [test_acc]
        # print(f"average train acc: {sum(train_acc_list) / len(train_acc_list)},"
        #       f"average test acc: {sum(test_acc_list) / len(test_acc_list)}")
        clustering_list = [client.identity_vec.tolist() for client in clients]
        cluster_identities = self.determine_cluster_index_using_kmeans(X=clustering_list, n_cluster=self.hp['K'])
        clients_ids = [client.id for client in clients]

        # === save client average embeddings ===
        print("Get features for three domain datasets")
        x_embeds = []
        for client in clients:
            # add domain#1 and domain#2 embeds
            x_embeds.extend(client.test_embeds)

        for client in clients:
            # add domain#0 embed
            x_embeds.extend(client.public_train_embeds)

        obj_path = "../data/avg_embed.obj"
        with open(obj_path, 'wb') as fp:
            pickle.dump(x_embeds, fp)
            print(f"save {obj_path}")

        # record client's (have seen) cluster identity
        for i in range(len(clients_ids)):
            client_id = clients_ids[i]
            identity = cluster_identities[i]
            self.client_identities[client_id] = identity

    def train(self, clients, lr):
        updates = []
        print("===Client Training Stage===")
        for client in clients:
            print(f"[client:#{client.id}, cluster:#{client.cluster_id}]")
            theta_all = client.local_training(lr=lr, iterations=self.hp['localTrainingEpoch'])
            updates += [(client.cluster_id, self.client_sizes[client.id], theta_all)]
            print("")
        self.updates = updates

    def aggregate(self):
        updates = self.updates

        group_weight_list = [0 for _ in range(self.hp['K'])]
        group_parameters_list = [deepcopy(self.models[i].state_dict()) for i in range(self.hp['K'])]

        for (identity, size, theta_all) in updates:
            if group_weight_list[identity] == 0:
                # first client
                for k in group_parameters_list[identity].keys():
                    group_parameters_list[identity][k] = size * theta_all[k]
            else:
                # later client
                for k in group_parameters_list[identity].keys():
                    group_parameters_list[identity][k] += size * theta_all[k]

            group_weight_list[identity] += size

        group_weight_iter = iter(group_weight_list)
        for parameters in group_parameters_list:
            w = next(group_weight_iter)
            if w == 0:
                # None of the currently selected clients belong to this cluster
                continue
            for k in parameters.keys():
                parameters[k] /= w

        # update server models
        for i in range(len(self.models)):
            self.models[i].load_state_dict(group_parameters_list[i])

    def test(self, clients):
        training_loss_list, test_loss_list = [], []
        training_acc_list, test_acc_list = [], []
        # pbar = tqdm(clients, ncols=100)
        print("===Client Validation Stage===")
        for client in clients:
            # pbar.set_description(f"Client#{client.id} is getting metrics")
            results_train = client.validation(loader=client.train_loader)
            results_test = client.validation(loader=client.test_loader)
            training_loss_list += [results_train['loss']]
            training_acc_list += [results_train['accuracy']]
            print(f"    client#{client.id}, train_acc:{results_train['accuracy']}, test_acc:{results_test['accuracy']}")

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

        print(f"IFCA performance, training loss: {average_training_loss}, "
              f"training acc: {average_training_acc}")
        print(f"IFCA performance, test loss: {average_test_loss}, "
              f"test acc: {average_test_acc}")

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
    def determine_cluster_index_using_kmeans(X, n_cluster=4):
        # clustering in terms of cosine similarity, if l2 pls comment the following line
        estimator = KMeans(n_clusters=n_cluster)
        estimator.fit(X)
        labels_pred = estimator.labels_
        return labels_pred
