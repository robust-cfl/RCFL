import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

import copy
import torch
import wandb
import random
import models.neural_nets as neural_nets
import data.data_utils as data_utils
from torch.utils.tensorboard import SummaryWriter
from algorithm.fedrep_client import Client
from algorithm.fedrep_server import Server
from utils.args import parse_args
from utils.utils import setup_seed
from prettytable import PrettyTable
from opacus.validators import ModuleValidator

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
    model = getattr(neural_nets, hp.model)()  # object
    ################################
    # domain-classifier requires-grad False
    for (key, param) in model.named_parameters():
        if key.startswith('domain'):
            param.requires_grad = False
    if hp.dp:
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=False)
    else:
        pass
    # object, need deepcopy
    clients = [Client(train_loader=train_loader, test_loader=test_loader, model=copy.deepcopy(model).to(device), hp=hp,
                      id_num=i) for i, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders))]

    server = Server(model=copy.deepcopy(model).to(device), hp=hp, stats=stats)

    # Print optimizer Specs
    print_model(D=clients[0])

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    for c_round in range(1, hp.communicationRounds + 1):
        print(f"Round {c_round}")
        # Update learning rate
        # exp decay
        updated_lr = hp.lr * hp.lrDecay ** (c_round // hp.decayStep)

        # server does
        server.broadcast(clients=clients)
        # test on training set and test set
        summary = server.test_on_test_data(clients=clients)
        server.train(clients=clients, lr=updated_lr)
        if hp.dp:
            summary.update({'epsilon': sum(server.epsilons)})
        server.aggregate()
        if hp.wandb:
            wandb.log(summary)
        else:
            for tag, value in summary.items():
                writer.add_scalar(tag, value, c_round)


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
