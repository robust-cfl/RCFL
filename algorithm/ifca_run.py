import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import copy
import torch
import wandb
import random
import pprint
import models.neural_nets as neural_nets
import data.data_utils as data_utils
from algorithm.ifca_client import Client
from algorithm.ifca_server import Server
from utils.args import parse_args
from utils.utils import setup_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(hp):
    """IFCA模型框架
    server上初始化K个模型
    第t轮：
    1. 选择全部client (集合M_t)
    2. 将当前server上的K个模型参数broadcast给M_t
    3. client利用自身训练集，选出K个模型中loss最小的模型，并记录对应的cluster id
        这一步直接测试下train和test的指标
    4. 更新client的参数为loss最小的参数，然后进行h轮本地训练
    5. 将训练好的模型参数和cluster id发送给server
    6. server在每个cluster内进行所有模型参数加权平均
    """
    # pprint.pprint(dict(hp))

    dataset = hp.dataset
    # load the data and split it among clients
    if dataset == 'digit5':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_digit5(hp=hp, verbose=True)
    elif dataset == 'office':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_office(hp=hp, verbose=True)
    elif dataset == 'home':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_office_home(hp=hp, verbose=True)
    else:
        assert dataset == 'pacs'
        train_loaders, test_loaders, public_loader, stats = data_utils.get_pacs(hp=hp, verbose=True)
    setup_seed(rs=hp.seed)

    # instantiate clients and server with neural net
    model = getattr(neural_nets, hp.model)()  # object
    # object, need deepcopy (每个客户端上只需要有一个能跑的模型)
    clients = [Client(train_loader=train_loader, test_loader=test_loader, model=copy.deepcopy(model).to(device), hp=hp,
                      id_num=i) for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders))]
    # 客户端上初始化K个不同模型 (K个cluster)
    server = Server(models=[getattr(neural_nets, hp.model)().to(device) for i in range(hp.K)],  # 模型参数需要不同
                    hp=hp,
                    stats=stats)

    # Print optimizer Specs
    print_model(D=clients[0])

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    for c_round in range(1, hp.communicationRounds + 1):
        print(f"Round {c_round}")
        participating_clients = clients
        # Update learning rate, exp decay
        updated_lr = hp.lr * hp.lrDecay ** (c_round // hp.decayStep)
        # 2. 将当前server上的K个模型参数broadcast个M_t
        # 3. 选出最好的模型和cluster_id，并计算训练集和测试集指标
        server.broadcast(clients=participating_clients, c_round=c_round,
                         stats=stats)
        summary = server.test(clients=participating_clients)
        wandb.log(summary)

        # 4. server发起train指令：clients进行classifyingEpoch轮本地训练
        server.train(clients=participating_clients, lr=updated_lr)
        # 5. 只用当前选中的客户端按cluster加权平均来更新cluster模型
        server.aggregate()


def print_model(D):
    print("Model {}:".format(D.hp.model))
    n = 0
    for key, value in D.model.named_parameters():
        print(' -', '{:30}'.format(key), list(value.shape))
        n += value.numel()
    print("Total number of Parameters: ", n)
    print()


if __name__ == '__main__':
    wandb.init(entity='xxx', project='xxx')

    args = parse_args()
    wandb.watch_called = False
    config = wandb.config
    config.update(args)

    run(hp=config)
