import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import torch
import pickle
import wandb
import models.neural_nets as neural_nets
import data.data_utils as data_utils
from torch.utils.tensorboard import SummaryWriter
from algorithm.fesem_client import Client
from algorithm.fesem_server import Server
from utils.args import parse_args
from utils.utils import setup_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(hp):
    print(hp)

    # load the data and split it among clients
    if hp.dataset == 'digit5':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_digit5(hp=hp, verbose=True)
    elif hp.dataset == 'office':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_office(hp=hp, verbose=True)
    elif hp.dataset == 'home':
        train_loaders, test_loaders, public_loader, stats = data_utils.get_office_home(hp=hp, verbose=True)
    else:
        assert hp.dataset == 'pacs'
        train_loaders, test_loaders, public_loader, stats = data_utils.get_pacs(hp=hp, verbose=True)
    setup_seed(rs=hp.seed)
    # instantiate clients and server with neural net
    model_class = getattr(neural_nets, hp.model)  # class
    ################################
    # class, dont need deepcopy
    clients = [Client(train_loader=train_loader, test_loader=test_loader, model_class=model_class, hp=hp,
                      id_num=i) for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders))]

    server = Server(hp=hp, stats=stats)

    # Print optimizer Specs
    print_model(D=clients[0])

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    identities_over_runs = {}
    for c_round in range(1, hp.communicationRounds + 1):
        print(f"\nRound {c_round}\n")
        # Update learning rate
        # exp decay
        updated_lr = hp.lr * hp.lrDecay ** (c_round // hp.decayStep)

        # server does
        server.update_centers(clients=clients)

        lst = [c.model_index for c in clients]
        identities_over_runs.update({c_round: lst})

        server.broadcast(clients=clients)
        # test on training set and test set
        summary = server.test_on_test_data(clients=clients)
        server.train(clients=clients, lr=updated_lr)
        if hp.dp:
            summary.update({'epsilon': sum(server.epsilons)})
        # server.aggregate()
        if hp.wandb:
            wandb.log(summary)
        else:
            print(summary)
    if hp.wandb:
        # 保存到wandb run文件夹下
        wandb_file_path = os.path.join(wandb.run.dir, "cluster_info_dict.obj")
        with open(wandb_file_path, 'wb') as fp:
            pickle.dump(identities_over_runs, fp)
        # 同步到云端
        wandb.save("cluster_info_dict.obj")
        # ============================================


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
        writer = SummaryWriter(f'./runs/{args.logname}')
        args.writer = writer
        config = args
    run(hp=config)
