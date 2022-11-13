import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import copy
import torch
import wandb
import random
import pickle
import models.neural_nets as neural_nets
import data.data_utils as data_utils
from torch.utils.tensorboard import SummaryWriter
from algorithm.client import Client
from algorithm.server import Server
from utils.args import parse_args
from utils.utils import setup_seed
from prettytable import PrettyTable
from opacus.validators import ModuleValidator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(hp):
    print(hp)
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
    if hp.dp:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
    else:
        # model = ModuleValidator.fix(model)
        # ModuleValidator.validate(model, strict=False)
        pass
    # object, need deepcopy
    clients = [Client(train_loader=train_loader, test_loader=test_loader, public_loader=public_loader,
                      model=copy.deepcopy(model).to(device), hp=hp,
                      id_num=i) for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders))]
    server = Server(public_loader=public_loader, models=[copy.deepcopy(model).to(device) for _ in range(hp.K)],
                    hp=hp, stats=stats)

    # Print optimizer Specs
    print_model(D=clients[0])

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    clustering_result = {}
    for c_round in range(1, hp.communicationRounds + 1):
        print(f"\nRound {c_round}")
        # Update learning rate
        # exp decay
        updated_lr = hp.lr * hp.lrDecay ** (c_round // hp.decayStep)

        # server does
        if c_round <= hp.warmupEpoch:
            # warm up stage
            server.fedavg_broadcast(clients=clients)
            summary = server.test_on_test_data(clients=clients)
            server.fedavg_train(clients=clients, lr=updated_lr)
            server.fedavg_aggregate()
        elif c_round <= hp.warmupEpoch + hp.clusteringWindowSize:
            # clustering stage
            server.broadcast(clients=clients)
            summary = server.test_on_test_data(clients=clients)
            server.train(clients=clients, lr=updated_lr, fixClusters=False)
            points, client_identities, round_clustering_result, purity = server.clustering(lambda_=hp.lambdA)
            # write clustering result, purity and upload it to wandb
            clustering_result.update({
                c_round: {
                    'points': points,
                    'info': round_clustering_result,
                    'purity': purity
                }
            })
            print("Current round clustering result:")
            print(round_clustering_result)
            print(f"Purity: {purity}")
            if config.wandb:
                wandb_file_path = os.path.join(wandb.run.dir, "cluster_info_dict.obj")
                with open(wandb_file_path, 'wb') as fp:
                    pickle.dump(clustering_result, fp)
                wandb.save("cluster_info_dict.obj")
            else:
                pass
            summary.update({'purity': purity})
            server.aggregate()
        else:
            # perform intra-cluster fedavg with the cluster identities fixed
            assert c_round > hp.warmupEpoch + hp.clusteringWindowSize
            server.broadcast(clients=clients)
            summary = server.test_on_test_data(clients=clients)
            server.train(clients=clients, lr=updated_lr, fixClusters=True)
            server.aggregate()
        if hp.dp:
            summary.update({'epsilon': sum(server.epsilons)})
        if config.wandb:
            wandb.log(summary)
        else:
            for tag, value in summary.items():
                config.writer.add_scalar(tag, value, c_round)


def print_model(D):
    print("Model {}:".format(D.hp.model))
    n = 0
    for key, value in D.model.named_parameters():
        print(' -', '{:30}'.format(key), list(value.shape))
        n += value.numel()
    print("Total number of Parameters: ", n)
    print()


if __name__ == '__main__':
    args = parse_args()
    if args.wandb:
        wandb.init(entity='xxx', project='xxx')
        wandb.watch_called = False
        config = wandb.config
        config.update(args)
    else:
        writer = SummaryWriter(f'./runs/{args.dataset}')
        args.writer = writer
        config = args
    run(hp=config)
