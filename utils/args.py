import argparse

ALGORITHMS = ['standalone', 'fedavg', 'fedprox', 'ifca', 'clusteredfl', 'fesem', 'perfedavg', 'fedrep', 'fedper', 'ditto', 'fedfomo', 'rcfl', 'entangled']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proxyCoefficient',
                        help='proxy coefficient for fedprox and fesem.',
                        type=float,
                        default=0.1)

    parser.add_argument('--wandb',
                        help='use wandb or tensorboard ?',
                        type=int,
                        default=0)

    parser.add_argument('--algorithm',
                        help='algorithm',
                        type=str,
                        default='fedavg',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('--dataset',
                        help='dataset',
                        type=str,
                        default='digit5',
                        choices=['digit5', 'office', 'pacs', 'home'])

    parser.add_argument('--model',
                        help='model',
                        type=str,
                        required=True)

    parser.add_argument('--domainAdaptationEpoch',
                        help='# of domain adaptation epochs',
                        type=int,
                        default=1)

    parser.add_argument('--communicationRounds',
                        help='# of communication rounds',
                        type=int,
                        default=10000)

    parser.add_argument('--nClients',
                        help='# of clients',
                        type=int,
                        default=100)

    parser.add_argument('--participationRate',
                        help='# of clients selected for federating',
                        type=float,
                        default=1.0)

    parser.add_argument('--batchSize',
                        help='batch size when clients train on local training data',
                        type=int,
                        default=50,
                        required=True)

    parser.add_argument('--logFrequency',
                        help='log frequency',
                        type=int,
                        default=1)

    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        default=3e-4)

    parser.add_argument('--lrDecay',
                        help='learning rate decay',
                        type=float,
                        default=1)

    parser.add_argument('--decayStep',
                        help='decay step',
                        type=int,
                        default=200)

    parser.add_argument('--optimizer',
                        help='optimizer',
                        type=str,
                        default='adam')

    parser.add_argument('--weightDecay',
                        help='weight decay',
                        type=float,
                        default=5e-4)

    parser.add_argument('--seed',
                        help='seed for randomly client sampling and batch splitting',
                        type=int,
                        default=24)

    parser.add_argument('--cuda',
                        help='using cuda ?',
                        type=int,
                        default=1)

    parser.add_argument('--K',
                        type=int,
                        help='assumed number of underlying distributions',
                        default=5)

    parser.add_argument('--ratio',
                        help='use partial dataset for training and test',
                        type=float,
                        default=1.0)

    parser.add_argument('--classifyingEpoch',
                        help='classifying epochs',
                        type=int,
                        default=1)

    parser.add_argument('--lambdA',
                        help='coefficient of the theta_i_vector',
                        type=float,
                        default=0.0)

    parser.add_argument('--mi',
                        help='coefficient of the mutual information loss',
                        type=float,
                        default=1.0)

    parser.add_argument('--isMode',
                        help='invariant feature encoder and specific feature encoder aggregation mode',
                        type=str,
                        choices=['add', 'concat'],
                        default='concat')

    parser.add_argument('--warmupEpoch',
                        help='warming up epochs',
                        type=int,
                        default=20)

    parser.add_argument('--clusteringWindowSize',
                        help='window size for clustering',
                        type=int,
                        default=5)

    parser.add_argument('--dp',
                        help='enable differential privacy ?',
                        type=int,
                        default=0)

    parser.add_argument("--sigma",
                        type=float,
                        default=1.0,
                        metavar="S",
                        help="Noise multiplier",
                        )

    parser.add_argument(
                        "-c",
                        "--max-per-sample-grad_norm",
                        type=float,
                        default=1.0,
                        metavar="C",
                        help="Clip per-sample gradients to this norm",
                        )

    parser.add_argument(
                        "--delta",
                        type=float,
                        default=1e-5,
                        metavar="D",
                        help="Target delta",
                        )

    return parser.parse_args()
