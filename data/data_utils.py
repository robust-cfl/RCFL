import os
import torch
import numpy as np
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from utils.utils import *
from pprint import pprint
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

DATA_PATH = '../data/raw_data'


class CustomImageDataset(Dataset):
    """
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    """

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.LongTensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.inputs.shape[0]


class PublicImageDataset(Dataset):
    """
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    """

    def __init__(self, public_X, public_Y, transforms=None):
        self.public_X = public_X
        self.public_Y = public_Y
        self.ids = [i for i in range(sum([y.shape[0] for y in self.public_Y]))]
        self.transforms = transforms
        self.cumsum = np.cumsum([y.shape[0] for y in self.public_Y])
        self.search_cumsum = self.cumsum - 1
        self.cumsum = np.hstack((0, self.cumsum))

    def __getitem__(self, index):
        sample_id = self.ids[index]
        which_dataset = np.searchsorted(self.search_cumsum, sample_id, side='left')
        index_intra_dataset = sample_id - self.cumsum[which_dataset]
        img, label = torch.tensor(self.public_X[which_dataset][index_intra_dataset], dtype=torch.float32), torch.tensor(
            self.public_Y[which_dataset][index_intra_dataset]).long()
        if self.transforms is not None:
            img = self.transforms[which_dataset](img)
        return img, label

    def __len__(self):
        return len(self.ids)


class OFFICE(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        label_name_list = os.listdir(self.root)
        self.label = []
        self.data = []

        for index, label_name in enumerate(label_name_list):
            label_name_2_index = {
                'mug': 0,
                'mouse': 1,
                'calculator': 2,
                'monitor': 3,
                'headphones': 4,
                'projector': 5,
                'laptop_computer': 6,
                'back_pack': 7,
                'backpack': 7,
                'keyboard': 8,
                'bike': 9
            }
            images_list = os.listdir(f"{self.root}/{label_name}")
            for img_name in images_list:
                img = Image.open(f"{root}/{label_name}/{img_name}").convert('RGB')
                img = np.array(img)
                self.label.append(label_name_2_index[label_name])
                if self.transform is not None:
                    img = self.transform(img)
                self.data.append(img)
        self.data = torch.stack(self.data)
        self.label = torch.tensor(self.label, dtype=torch.long)

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, mnist_root=os.path.join(DATA_PATH, "MNIST"), train=True,
                 transform=None, target_transform=None, download=False):
        """
        Init MNIST-M dataset.
        :param root: MNISTM dataset saved path
        :param mnist_root: MNIST dataset saved path
        :param train:
        :param transform:
        :param target_transform:
        :param download:
        """
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file)
            )
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file)
            )

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.url)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print("Processing...")

        # load MNIST-M images from pkl file
        with open(file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b"train"])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b"test"])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root, train=True, download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root, train=False, download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


class SyntheticDigits(VisionDataset):
    """Synthetic Digits Dataset.
    Code: https://github.com/liyxi/synthetic-digits/blob/main/synthetic_digits.py
    """

    resources = [
        ('http://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('http://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init Synthetic Digits dataset."""
        super(SyntheticDigits, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the Synthetic Digits data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class PACS(Dataset):
    def __init__(self, root_path, domain, train=True, transform=None, target_transform=None):
        self.root = f"{root_path}/{domain}"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        label_name_list = os.listdir(self.root)
        self.label = []
        self.data = []

        if not os.path.exists(f"{root_path}/precessed"):
            os.makedirs(f"{root_path}/precessed")
        if os.path.exists(f"{root_path}/precessed/{domain}_data.pt") and os.path.exists(
                f"{root_path}/precessed/{domain}_label.pt"):
            print(f"Load {domain} data and label from cache.")
            self.data = torch.load(f"{root_path}/precessed/{domain}_data.pt")
            self.label = torch.load(f"{root_path}/precessed/{domain}_label.pt")
        else:
            print(f"Getting {domain} datasets")
            for index, label_name in enumerate(label_name_list):
                label_name_2_index = {
                    'dog': 0,
                    'elephant': 1,
                    'giraffe': 2,
                    'guitar': 3,
                    'horse': 4,
                    'house': 5,
                    'person': 6,
                }
                images_list = os.listdir(f"{self.root}/{label_name}")
                for img_name in images_list:
                    img = Image.open(f"{self.root}/{label_name}/{img_name}").convert('RGB')
                    img = np.array(img)
                    self.label.append(label_name_2_index[label_name])
                    if self.transform is not None:
                        img = self.transform(img)
                    self.data.append(img)
            self.data = torch.stack(self.data)
            self.label = torch.tensor(self.label, dtype=torch.long)
            torch.save(self.data, f"{root_path}/precessed/{domain}_data.pt")
            torch.save(self.label, f"{root_path}/precessed/{domain}_label.pt")

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_synthetic():
    data_train = SyntheticDigits(root=os.path.join(DATA_PATH, "SYNTHETIC"),
                                 train=True, download=True)

    data_test = SyntheticDigits(root=os.path.join(DATA_PATH, "SYNTHETIC"),
                                train=False, download=True)

    # plotPics(data_train, 'synthetic', 3, 4, offset=2, filename='synthetic')

    x_train, y_train = data_train.data.numpy().transpose((0, 3, 1, 2)) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.numpy().transpose((0, 3, 1, 2)) / 255, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_mnistm():
    """Return MNISTM train/test data and labels as numpy arrays"""
    data_train = MNISTM(root=os.path.join(DATA_PATH, "MNISTM"),
                        mnist_root=os.path.join(DATA_PATH, "MNIST"),
                        train=True, download=True)

    data_test = MNISTM(root=os.path.join(DATA_PATH, "MNISTM"),
                       mnist_root=os.path.join(DATA_PATH, "MNIST"),
                       train=False, download=True)

    # plotPics(data_train, 'mnistm', 3, 4, offset=2, filename='mnistm')

    x_train, y_train = data_train.train_data.numpy().transpose((0, 3, 1, 2)) / 255, np.array(data_train.train_labels)
    x_test, y_test = data_test.test_data.numpy().transpose((0, 3, 1, 2)) / 255, np.array(data_test.test_labels)

    return x_train, y_train, x_test, y_test


def get_usps():
    """Return USPS train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.USPS(root=os.path.join(DATA_PATH, "USPS"), train=True, download=True)
    data_test = torchvision.datasets.USPS(root=os.path.join(DATA_PATH, "USPS"), train=False, download=True)

    x_train, y_train = data_train.data.reshape(-1, 1, 16, 16) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.reshape(-1, 1, 16, 16) / 255, np.array(data_test.targets)

    # plotPics((x_train, y_train), 'usps', 3, 3, offset=23, filename='usps')

    return x_train, y_train, x_test, y_test


def get_svhn():
    """Return SVHN train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.SVHN(root=os.path.join(DATA_PATH, "SVHN"), split='train', download=True)
    data_test = torchvision.datasets.SVHN(root=os.path.join(DATA_PATH, "SVHN"), split='test', download=True)

    # plotPics(data_train, 'svhn', 3, 4, offset=15, filename='svhn')

    x_train, y_train = data_train.data / 255, np.array(data_train.labels)
    x_test, y_test = data_test.data / 255, np.array(data_test.labels)
    return x_train, y_train, x_test, y_test


def get_mnist():
    """Return MNIST train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=True, download=True)
    data_test = torchvision.datasets.MNIST(root=os.path.join(DATA_PATH, "MNIST"), train=False, download=True)

    x_train, y_train = data_train.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(data_test.targets)

    # plotPics((x_train, y_train), 'mnist', 3, 3, offset=15, filename='mnist')

    return x_train, y_train, x_test, y_test


def get_cifar10():
    """Return CIFAR10 train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=True, download=True)
    data_test = torchvision.datasets.CIFAR10(root=os.path.join(DATA_PATH, "CIFAR10"), train=False, download=True)

    # plotPics(data_train, 'mnistm', 3, 4, offset=2, filename='mnistm')

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)) / 255, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_cifar100():
    """Return CIFAR100 train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.CIFAR100(root=os.path.join(DATA_PATH, "CIFAR100"), train=True, download=True)
    data_test = torchvision.datasets.CIFAR100(root=os.path.join(DATA_PATH, "CIFAR100"), train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)) / 255, np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)) / 255, np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def get_default_data_transforms(name):
    transforms_train = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'mnistm': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'usps': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'svhn': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'synthetic': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'office': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        # cifar100 refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b/utils.py#L166
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    }

    transforms_eval = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'mnistm': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'usps': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'svhn': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'synthetic': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'office': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    }

    return transforms_train[name], transforms_eval[name]


def split_image_data(data, labels, n_clients=100, classes_per_client=-10, shuffle=True, verbose=True):
    """
    Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
    different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    """
    # constants
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    data_per_client = [n_data // n_clients] * n_clients
    data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients

    if sum(data_per_client) > n_data:
        print("Impossible Split")
        exit()

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    return clients_split


def split_cifar100(x, y, rand_set_all=[]):
    # fine to coarse labels
    # number of clients, N = 100
    N = 100
    clients_split = []

    def fine2coarse(targets):
        """Convert Pytorch CIFAR100 fine targets to coarse targets."""
        coarse_2_fine_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                         3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                         6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                         0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                         5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                         16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                         10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                         2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                         16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                         18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        return coarse_2_fine_labels[targets]

    coarse_labels = fine2coarse(y)
    coarse_group_ids = {i: [] for i in range(20)}
    for i in range(len(y)):
        coarse_l = coarse_labels[i]
        coarse_group_ids[coarse_l] += [i]
    for user_id in range(N):
        group_id = int(user_id / 5)
        drift = user_id % 5
        client_size = int(len(coarse_group_ids[group_id]) / 5)
        left = len(coarse_group_ids[group_id]) - 5 * client_size
        client_ids = coarse_group_ids[group_id][drift * client_size: (drift + 1) * client_size]
        if drift < left:
            client_ids += coarse_group_ids[group_id][-left + drift]
        clients_split += [(x[client_ids], y[client_ids])]
    return clients_split


def split_dataset_iid(x, y, verbose=False, train=True):
    # number of clients per dataset, M = 50
    M = 5
    N = y.shape[0]
    clients_split = []
    ids = [i for i in range(N)]
    np.random.shuffle(ids)
    # samples_per_client = int(N * hp['ratio'] // M)
    if train:
        samples_per_client = 1166
    else:
        samples_per_client = 401
    if verbose:
        print(f"{N} clients, samples per client {samples_per_client}.")
    for i in range(M):
        ids_tmp = ids[i * samples_per_client: (i + 1) * samples_per_client]
        x_tmp = x[ids_tmp]
        y_tmp = y[ids_tmp]
        clients_split += [(x_tmp, y_tmp)]
    return clients_split


def split_dataset_iid_pacs(x, y, M=None, verbose=False, train=True):
    """
    :param M: 客户端数量
    """
    N = y.shape[0]
    if train:
        samples_per_client = 500
    else:
        samples_per_client = int(N / M)
    clients_split = []
    ids = [i for i in range(N)]
    np.random.shuffle(ids)
    # samples_per_client = int(N * hp['ratio'] // M)
    for i in range(M):
        ids_tmp = ids[i * samples_per_client: (i + 1) * samples_per_client]
        x_tmp = x[ids_tmp]
        y_tmp = y[ids_tmp]
        clients_split += [(x_tmp, y_tmp)]

    return clients_split


def split_mnist(x, y, small_dataset_ratio=0.0, verbose=False):
    # number of clients, N = 100
    N = 100
    clients_split = []

    # split a small dataset allocated to all the clients evenly
    group_ids = {i: [] for i in range(10)}
    for i in range(len(y)):
        group_ids[y[i]] += [i]
    small_dataset_ids = []
    for key, val in group_ids.items():
        size = int(len(val) * small_dataset_ratio)
        small_dataset_ids += val[0: size]
        group_ids.update({key: val[size:]})

    random.shuffle(small_dataset_ids)

    # combine 5 classes to a "superclass", i.e., a underlying distribution
    small_dataset_split_size = int(len(small_dataset_ids) / 2)
    cluster_ids = {}
    for key, val in group_ids.items():
        # key == 0
        if key % 5 == 0:
            cluster_index = int(key / 5)
            cluster_ids.update({cluster_index: group_ids[key] + group_ids[key + 1] + group_ids[key + 2] + group_ids[
                key + 3] + group_ids[key + 4] + small_dataset_ids[
                                                cluster_index * small_dataset_split_size: (
                                                                                                  cluster_index + 1) * small_dataset_split_size]})
            random.shuffle(cluster_ids[cluster_index])
        else:
            pass

    if verbose:
        # print clusters' distribution
        for key, val in cluster_ids.items():
            labels_distribution = [0] * 10
            for item in val:
                labels_distribution[y[item]] += 1
            # print(f"cluster {key}", labels_distribution)

    users_identity = []
    for user_id in range(N):
        # split 100 clients into 2 groups, and one group(50 clients) share all the 5 classes samples.
        group_id = int(user_id / 50)
        users_identity += [group_id]
        drift = user_id % 50
        client_size = int(len(cluster_ids[group_id]) / 50)
        left = len(cluster_ids[group_id]) - 50 * client_size
        client_ids = cluster_ids[group_id][drift * client_size: (drift + 1) * client_size]
        if drift < left:
            client_ids += [cluster_ids[group_id][-left + drift]]
        clients_split += [(x[client_ids], y[client_ids])]
    return clients_split, users_identity


def split_cifar10(x, y, small_dataset_ratio=0.0, verbose=False):
    # number of clients, N = 100
    N = 100
    clients_split = []

    # split a small dataset allocated to all the clients evenly
    group_ids = {i: [] for i in range(10)}
    for i in range(len(y)):
        group_ids[y[i]] += [i]
    small_dataset_ids = []
    for key, val in group_ids.items():
        size = int(len(val) * small_dataset_ratio)
        small_dataset_ids += val[0: size]
        group_ids.update({key: val[size:]})

    random.shuffle(small_dataset_ids)

    # combine two classes to a "superclass", i.e., a underlying distribution
    small_dataset_split_size = int(len(small_dataset_ids) / 5)
    cluster_ids = {}
    for key, val in group_ids.items():
        # key == 0
        if key % 2 == 0:
            cluster_index = int(key / 2)
            cluster_ids.update({cluster_index: group_ids[key] + group_ids[key + 1] + small_dataset_ids[
                                                                                     cluster_index * small_dataset_split_size: (
                                                                                                                                       cluster_index + 1) * small_dataset_split_size]})
            random.shuffle(cluster_ids[cluster_index])
        else:
            pass

    if verbose:
        # print clusters' distribution
        for key, val in cluster_ids.items():
            labels_distribution = [0] * 10
            for item in val:
                labels_distribution[y[item]] += 1
            # print(f"cluster {key}", labels_distribution)

    users_identity = []
    for user_id in range(N):
        # split 100 clients into 5 groups, and one group(20 clients) share all the two classes samples.
        group_id = int(user_id / 20)
        users_identity += [group_id]
        drift = user_id % 20
        client_size = int(len(cluster_ids[group_id]) / 20)
        left = len(cluster_ids[group_id]) - 20 * client_size
        client_ids = cluster_ids[group_id][drift * client_size: (drift + 1) * client_size]
        if drift < left:
            client_ids += [cluster_ids[group_id][-left + drift]]
        clients_split += [(x[client_ids], y[client_ids])]
    return clients_split, users_identity


def dirichlet_partition(users_num=100, alpha=0.5, x_train=None, y_train=None, x_test=None, y_test=None, verbose=True,
                        clusters_num=5):
    clients_train_split = []
    clients_test_split = []
    num_of_classes = np.unique(y_train).shape[0]
    for k in range(num_of_classes):
        train_ids, test_ids = np.where(y_train == k)[0], np.where(y_test == k)[0]
        np.random.shuffle(train_ids)
        np.random.shuffle(test_ids)

        proportions = np.random.dirichlet(np.repeat(alpha, users_num))
        train_batch_sizes = [int(p * len(train_ids)) for p in proportions]
        test_batch_sizes = [int(p * len(test_ids)) for p in proportions]

        train_start = 0
        test_start = 0

        for i in range(users_num):
            train_size = train_batch_sizes[i]
            test_size = test_batch_sizes[i]

            tn_ids = train_ids[train_start: train_start + train_size].tolist()
            tt_ids = test_ids[test_start: test_start + test_size].tolist()
            if len(clients_train_split) >= users_num:
                clients_train_split[i] = (np.append(clients_train_split[i][0], x_train[tn_ids], axis=0),
                                          np.append(clients_train_split[i][1], y_train[tn_ids], axis=0))
                clients_test_split[i] = (np.append(clients_test_split[i][0], x_test[tt_ids], axis=0),
                                         np.append(clients_test_split[i][1], y_test[tt_ids], axis=0))
            else:
                clients_train_split += [(x_train[tn_ids], y_train[tn_ids])]
                clients_test_split += [(x_test[tt_ids], y_test[tt_ids])]

    def get_labels_distribution(train_split, test_split):
        train_labels_distribution = {}
        for c_id, (x, y) in enumerate(train_split):
            dis = [len(np.where(y == label)[0]) / len(y) for label in range(num_of_classes)]
            train_labels_distribution.update({c_id: dis})

        test_labels_distribution = {}
        for c_id, (x, y) in enumerate(test_split):
            dis = [len(np.where(y == label)[0]) / len(y) for label in range(num_of_classes)]
            test_labels_distribution.update({c_id: dis})
        return train_labels_distribution, test_labels_distribution

    train_dis, test_dis = get_labels_distribution(clients_train_split, clients_test_split)
    if verbose:
        pprint(train_dis)
        pprint(test_dis)

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

    def get_similarity_matrix(distribution):
        N = len(distribution.keys())
        M = [0] * (N * N)
        for client_1, dis_1 in distribution.items():
            for client_2, dis_2 in distribution.items():
                sim = cosine_similarity(np.array(dis_1), np.array(dis_2))
                M[client_1 * N + client_2] = sim
        return M

    def get_emd_matrix(distribution):
        N = len(distribution.keys())
        M = [0] * (N * N)
        for client_1, dis_1 in distribution.items():
            for client_2, dis_2 in distribution.items():
                d = np.linalg.norm(np.array(dis_1) - np.array(dis_2), ord=1)
                M[client_1 * N + client_2] = d
        return M

    # similarity_matrix = get_similarity_matrix(train_dis)
    emd_matrix = get_emd_matrix(train_dis)

    def determine_cluster_index(matrix, c_num=5, reverse=True):
        N = int(np.sqrt(len(matrix)))
        clusters = {cluster_index: set([cluster_index]) for cluster_index in range(N)}
        sorted_m = sorted(matrix, reverse=reverse)
        for sim in sorted_m:
            index = matrix.index(sim)
            matrix[index] = -1
            client_1 = index // N
            client_2 = index % N

            c_tmp = set()

            for c_index in list(clusters.keys()):
                c = clusters[c_index]
                if client_1 in c or client_2 in c:
                    c_tmp = c_tmp.union(c)
                    clusters.__delitem__(c_index)
            clusters.update({max(clusters.keys()) + 1: c_tmp})
            # print(len(clusters))
            if len(clusters) == c_num:
                print("Satisfying clusters number requirements!")
                return clusters
        print("Error! Don't satisfy clusters number requirements!")
        return clusters

    def determine_cluster_index_using_kmeans(dis, n_cluster=4):
        dis_array = [dis[k] for k in range(len(dis))]
        estimator = KMeans(n_clusters=n_cluster)
        estimator.fit(dis_array)
        labels_pred = estimator.labels_
        return labels_pred

    users_cluster_identity = determine_cluster_index_using_kmeans(dis=train_dis, n_cluster=clusters_num)

    # m_clusters = determine_cluster_index(matrix=emd_matrix, c_num=clusters_num, reverse=False)
    #
    # print(m_clusters)
    # final_clusters = {}
    # for step, key in enumerate(m_clusters):
    #     final_clusters.update({step: m_clusters[key]})
    # print(final_clusters)
    #
    # def get_cluster_index(user_id):
    #     for cluster_index, val in final_clusters.items():
    #         if user_id in val:
    #             return cluster_index
    #     print("Didn't find the corresponding the cluster index!")
    #     return -1

    # users_cluster_identity = [get_cluster_index(i) for i in range(users_num)]

    # return clients_train_split, clients_test_split, users_cluster_identity, final_clusters
    return clients_train_split, clients_test_split, users_cluster_identity


def get_dataset_stats(y_train, y_test):
    print(f"# of training samples: {y_train.shape[0]}, # of test samples: {y_test.shape[0]}")

    def stats(y):
        num_of_classes = np.unique(y).shape[0]
        summary = {i: [] for i in range(num_of_classes)}
        for i in range(len(y)):
            summary.update({y[i]: summary[y[i]] + [i]})
        return summary

    for key, val in stats(y_train).items():
        print(f"{key}: {len(val)}", end=' ')
    for key, val in stats(y_train).items():
        print(f"{key}: {len(val)}", end=' ')
    print()


def split_public_dataset(x, y, ratio=0.2, remain=0, use_ratio=True):
    """Return public dataset and the left data for supervised training"""
    print("Split public unlabelled auxiliary dataset...")
    N = y.shape[0]
    all_ids = [i for i in range(N)]
    random.shuffle(all_ids)
    if use_ratio:
        public_samples_N = int(N * ratio)
    else:
        public_samples_N = remain
    public_ids = all_ids[0: public_samples_N]
    train_ids = all_ids[public_samples_N:]
    x_public, y_public = x[public_ids], y[public_ids]
    x_train, y_train = x[train_ids], y[train_ids]
    return x_public, y_public, x_train, y_train


def get_digit5(hp, verbose=False):
    setup_seed(rs=24)
    # Dataset
    mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = get_mnist()
    mnistm_x_train, mnistm_y_train, mnistm_x_test, mnistm_y_test = get_mnistm()
    svhn_x_train, svhn_y_train, svhn_x_test, svhn_y_test = get_svhn()
    usps_x_train, usps_y_train, usps_x_test, usps_y_test = get_usps()
    synthetic_x_train, synthetic_y_train, synthetic_x_test, synthetic_y_test = get_synthetic()

    if verbose:
        print(
            f"MNIST: # of training samples: {mnist_x_train.shape[0]}, # of test samples: {mnist_x_test.shape[0]}")

        print(
            f"MNIST-M: # of training samples: {mnistm_x_train.shape[0]}, # of test samples: {mnistm_x_test.shape[0]}")

        print(
            f"SVHN: # of training samples: {svhn_x_train.shape[0]}, # of test samples: {svhn_x_test.shape[0]}")

        print(
            f"USPS: # of training samples: {usps_x_train.shape[0]}, # of test samples: {usps_x_test.shape[0]}")

        print(
            f"Synthetic: # of training samples: {synthetic_x_train.shape[0]}, # of test samples: {synthetic_x_test.shape[0]}")

    # split public unlabelled auxiliary dataset
    mnist_x_public, mnist_y_public, mnist_x_train, mnist_y_train = split_public_dataset(x=mnist_x_train,
                                                                                        y=mnist_y_train, ratio=0.2)
    mnist_x_public, mnist_y_public = mnist_x_public[0: 12000], mnist_y_public[0: 12000]
    mnistm_x_public, mnistm_y_public, mnistm_x_train, mnistm_y_train = split_public_dataset(x=mnistm_x_train,
                                                                                            y=mnistm_y_train, ratio=0.2)
    mnistm_x_public, mnistm_y_public = mnistm_x_public[0: 12000], mnistm_y_public[0: 12000]
    svhn_x_public, svhn_y_public, svhn_x_train, svhn_y_train = split_public_dataset(x=svhn_x_train, y=svhn_y_train,
                                                                                    ratio=0.2)
    svhn_x_public, svhn_y_public = svhn_x_public[0: 12000], svhn_y_public[0: 12000]
    usps_x_public, usps_y_public, usps_x_train, usps_y_train = split_public_dataset(x=usps_x_train, y=usps_y_train,
                                                                                    ratio=0.2)
    synthetic_x_public, synthetic_y_public, synthetic_x_train, synthetic_y_train = split_public_dataset(
        x=synthetic_x_train, y=synthetic_y_train, ratio=0.2)
    synthetic_x_public, synthetic_y_public = synthetic_x_public[0: 12000], synthetic_y_public[0: 12000]
    public_X = [mnist_x_public, mnistm_x_public, svhn_x_public, usps_x_public, synthetic_x_public]
    public_Y = [mnist_y_public, mnistm_y_public, svhn_y_public, usps_y_public, synthetic_y_public]

    # Transforms
    mnist_transforms_train, mnist_transforms_eval = get_default_data_transforms('mnist')
    mnistm_transforms_train, mnistm_transforms_eval = get_default_data_transforms('mnistm')
    svhn_transforms_train, svhn_transforms_eval = get_default_data_transforms('svhn')
    usps_transforms_train, usps_transforms_eval = get_default_data_transforms('usps')
    synthetic_transforms_train, synthetic_transforms_eval = get_default_data_transforms('synthetic')
    # cifar10_transforms_train, cifar10_transforms_eval = get_default_data_transforms('cifar10')

    if verbose:
        print(
            f"MNIST: # of training samples: {mnist_x_train.shape[0]}, # of test samples: {mnist_x_test.shape[0]}")

        print(
            f"MNIST-M: # of training samples: {mnistm_x_train.shape[0]}, # of test samples: {mnistm_x_test.shape[0]}")

        print(
            f"SVHN: # of training samples: {svhn_x_train.shape[0]}, # of test samples: {svhn_x_test.shape[0]}")

        print(
            f"USPS: # of training samples: {usps_x_train.shape[0]}, # of test samples: {usps_x_test.shape[0]}")

        print(
            f"Synthetic: # of training samples: {synthetic_x_train.shape[0]}, # of test samples: {synthetic_x_test.shape[0]}")

    ################################################################################
    # SVHN ## SVHN ## SVHN ## SVHN ## SVHN ## SVHN ## SVHN ## SVHN ## SVHN ## SVHN #
    ################################################################################
    svhn_train_splits = split_dataset_iid(x=svhn_x_train, y=svhn_y_train, train=True)
    svhn_test_splits = split_dataset_iid(x=svhn_x_test, y=svhn_y_test, train=False)
    svhn_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=svhn_transforms_train),
                   batch_size=hp.batchSize, shuffle=True) for x, y in svhn_train_splits]
    svhn_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=svhn_transforms_eval),
                   batch_size=hp.batchSize, shuffle=False) for x, y in svhn_test_splits]

    #################################################################################
    # MNIST ## MNIST ## MNIST ## MNIST ## MNIST ## MNIST ## MNIST ## MNIST ## MNIST #
    #################################################################################
    mnist_train_splits = split_dataset_iid(x=mnist_x_train, y=mnist_y_train, train=True)
    mnist_test_splits = split_dataset_iid(x=mnist_x_test, y=mnist_y_test, train=False)

    mnist_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=mnist_transforms_train),
                   batch_size=hp.batchSize, shuffle=True) for x, y in mnist_train_splits]
    mnist_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=mnist_transforms_eval),
                   batch_size=hp.batchSize, shuffle=False) for x, y in mnist_test_splits]

    #############################################################################
    # MNIST-M ## MNIST-M ## MNIST-M ## MNIST-M ## MNIST-M ## MNIST-M ## MNIST-M #
    #############################################################################
    mnistm_train_splits = split_dataset_iid(x=mnistm_x_train, y=mnistm_y_train, train=True)
    mnistm_test_splits = split_dataset_iid(x=mnistm_x_test, y=mnistm_y_test, train=False)

    mnistm_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=mnistm_transforms_train),
                   batch_size=hp.batchSize, shuffle=True) for x, y in mnistm_train_splits]
    mnistm_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=mnistm_transforms_eval),
                   batch_size=hp.batchSize, shuffle=False) for x, y in mnistm_test_splits]

    ########################################################################
    # USPS ## USPS ## USPS ## USPS ## USPS ## USPS ## USPS ## USPS ## USPS #
    ########################################################################
    usps_train_splits = split_dataset_iid(x=usps_x_train, y=usps_y_train, train=True)
    usps_test_splits = split_dataset_iid(x=usps_x_test, y=usps_y_test, train=False)

    usps_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=usps_transforms_train),
                   batch_size=hp.batchSize, shuffle=True) for x, y in usps_train_splits]
    usps_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=usps_transforms_eval),
                   batch_size=hp.batchSize, shuffle=False) for x, y in usps_test_splits]

    ##############################################################################
    # Synthetic ## Synthetic ## Synthetic ## Synthetic ## Synthetic ## Synthetic #
    ##############################################################################
    synthetic_train_splits = split_dataset_iid(x=synthetic_x_train, y=synthetic_y_train)
    synthetic_test_splits = split_dataset_iid(x=synthetic_x_test, y=synthetic_y_test)

    synthetic_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=synthetic_transforms_train),
                   batch_size=hp.batchSize, shuffle=True) for x, y in synthetic_train_splits]
    synthetic_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=synthetic_transforms_eval),
                   batch_size=hp.batchSize, shuffle=False) for x, y in synthetic_test_splits]

    ##################
    #     PUBLIC     #
    ##################
    def collate_fn(data):
        data_list = []
        label_list = []
        for j in range(0, len(data)):
            img, label = data[j][0], data[j][1]
            if img.shape[0] < 3:
                img = img.expand(3, img.shape[1], img.shape[2])
            data_list.append(img)
            label_list.append(label)
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        data_copy = data_tensor, label_tensor
        return data_copy

    public_transforms = [mnist_transforms_train, mnistm_transforms_train, svhn_transforms_train, usps_transforms_train,
                         synthetic_transforms_train]
    public_loader = DataLoader(
        PublicImageDataset(public_X=public_X, public_Y=public_Y, transforms=public_transforms),
        batch_size=hp.batchSize, shuffle=True, collate_fn=collate_fn)

    train_loaders = []
    test_loaders = []
    train_loaders += mnist_train_loaders
    train_loaders += mnistm_train_loaders
    train_loaders += svhn_train_loaders
    train_loaders += usps_train_loaders
    train_loaders += synthetic_train_loaders

    test_loaders += mnist_test_loaders
    test_loaders += mnistm_test_loaders
    test_loaders += svhn_test_loaders
    test_loaders += usps_test_loaders
    test_loaders += synthetic_test_loaders

    CIS = []
    CIS += ['mnist'] * len(mnist_train_loaders)
    CIS += ['mnistm'] * len(mnistm_train_loaders)
    CIS += ['svhn'] * len(svhn_train_loaders)
    CIS += ['usps'] * len(usps_train_loaders)
    CIS += ['synthetic'] * len(synthetic_train_loaders)

    stats = {"split": [loader.sampler.num_samples for loader in train_loaders],
             "test_split": [loader.dataset.labels.shape[0] for loader in test_loaders],
             "client_identities": CIS}

    return train_loaders, test_loaders, public_loader, stats


def partition_office(dataset):
    data, label = dataset.data, dataset.label
    clients_num = len(label) // 100
    public_len = len(label) % 100
    ids = [i for i in range(len(label))]
    np.random.shuffle(ids)
    data = data[ids]
    label = label[ids]
    train_splits, test_splits, public_split = [], [], []
    public_split += [(data[-public_len:], label[-public_len:])]
    for j in range(clients_num):
        client_ids = ids[100 * j: 100 * (j + 1)]
        client_data, client_label = data[client_ids], label[client_ids]
        train_splits += [(client_data[0: 80], client_label[0: 80])]
        test_splits += [(client_data[80:], client_label[80:])]
    return train_splits, test_splits, public_split


def get_office(hp, verbose=False):
    setup_seed(rs=24)
    train_transform, eval_transform = get_default_data_transforms('office')
    amazon = OFFICE(root=os.path.join(DATA_PATH, 'OFFICE/amazon'), transform=train_transform)
    caltech = OFFICE(root=os.path.join(DATA_PATH, 'OFFICE/caltech'), transform=train_transform)
    dslr = OFFICE(root=os.path.join(DATA_PATH, 'OFFICE/dslr'), transform=train_transform)
    webcam = OFFICE(root=os.path.join(DATA_PATH, 'OFFICE/webcam'), transform=train_transform)

    amazon_train_splits, amazon_test_splits, amazon_public_split = partition_office(amazon)
    caltech_train_splits, caltech_test_splits, caltech_public_split = partition_office(caltech)
    dslr_train_splits, dslr_test_splits, dslr_public_split = partition_office(dslr)
    webcam_train_splits, webcam_test_splits, webcam_public_split = partition_office(webcam)

    amazon_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=train_transform),
                   batch_size=hp.batchSize, shuffle=True) for x, y in amazon_train_splits]
    amazon_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=eval_transform),
                   batch_size=hp.batchSize, shuffle=False) for x, y in amazon_test_splits]

    caltech_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=train_transform),
                   batch_size=hp.batchSize, shuffle=True) for x, y in caltech_train_splits]
    caltech_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=eval_transform),
                   batch_size=hp.batchSize, shuffle=False) for x, y in caltech_test_splits]

    dslr_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=train_transform),
                   batch_size=hp.batchSize, shuffle=True) for x, y in dslr_train_splits]
    dslr_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=eval_transform),
                   batch_size=hp.batchSize, shuffle=False) for x, y in dslr_test_splits]

    webcam_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=train_transform),
                   batch_size=hp.batchSize, shuffle=True) for x, y in webcam_train_splits]
    webcam_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, transforms=eval_transform),
                   batch_size=hp.batchSize, shuffle=False) for x, y in webcam_test_splits]

    public_data, public_label = [], []
    for split in amazon_public_split:
        public_data += split[0]
        public_label += split[1]
    for split in caltech_public_split:
        public_data += split[0]
        public_label += split[1]
    for split in dslr_public_split:
        public_data += split[0]
        public_label += split[1]
    for split in webcam_public_split:
        public_data += split[0]
        public_label += split[1]
    public_data, public_label = torch.stack(public_data), torch.tensor(public_label, dtype=torch.long)

    public_transform = train_transform
    public_loader = DataLoader(
        CustomImageDataset(inputs=public_data, labels=public_label, transforms=public_transform),
        batch_size=hp.batchSize, shuffle=True)

    train_loaders = []
    test_loaders = []
    train_loaders += amazon_train_loaders
    train_loaders += caltech_train_loaders
    train_loaders += dslr_train_loaders
    train_loaders += webcam_train_loaders

    test_loaders += amazon_test_loaders
    test_loaders += caltech_test_loaders
    test_loaders += dslr_test_loaders
    test_loaders += webcam_test_loaders

    CIS = []
    CIS += [0] * len(amazon_train_splits)
    CIS += [1] * len(caltech_train_splits)
    CIS += [2] * len(dslr_train_splits)
    CIS += [3] * len(webcam_train_splits)

    stats = {"split": [loader.sampler.num_samples for loader in train_loaders],
             "test_split": [loader.dataset.labels.shape[0] for loader in test_loaders],
             "client_identities": CIS}

    return train_loaders, test_loaders, public_loader, stats


def get_pacs_domain(root_path=f"{DATA_PATH}/PACS", domain='art_painting', verbose=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    all_data = PACS(root_path, domain, transform=transform)
    # train:test=8:2
    x_train, x_test, y_train, y_test = train_test_split(all_data.data.numpy(), all_data.label.numpy(),
                                                        test_size=0.20, random_state=42)

    # if verbose:
        # plotPics((x_train, y_train), 'pacs', 3, 3, offset=150, filename=f'{domain}.pdf')

    return x_train, y_train, x_test, y_test


def get_pacs(hp, verbose=False):
    setup_seed(rs=24)
    # ==== PACS
    art_painting_x_train, art_painting_y_train, art_painting_x_test, art_painting_y_test = get_pacs_domain(
        domain='art_painting', verbose=False)
    print(
        f"art_painting: # of training samples: {art_painting_x_train.shape[0]}, # of test samples: {art_painting_x_test.shape[0]}")
    cartoon_x_train, cartoon_y_train, cartoon_x_test, cartoon_y_test = get_pacs_domain(domain='cartoon',
                                                                                       verbose=False)
    print(
        f"cartoon: # of training samples: {cartoon_x_train.shape[0]}, # of test samples: {cartoon_x_test.shape[0]}")
    photo_x_train, photo_y_train, photo_x_test, photo_y_test = get_pacs_domain(domain='photo', verbose=False)
    print(
        f"photo: # of training samples: {photo_x_train.shape[0]}, # of test samples: {photo_x_test.shape[0]}")
    sketch_x_train, sketch_y_train, sketch_x_test, sketch_y_test = get_pacs_domain(domain='sketch', verbose=False)
    print(
        f"sketch: # of training samples: {sketch_x_train.shape[0]}, # of test samples: {sketch_x_test.shape[0]}")

    # split public unlabelled auxiliary dataset
    art_painting_x_public, art_painting_y_public, art_painting_x_train, art_painting_y_train = split_public_dataset(
        x=art_painting_x_train,
        y=art_painting_y_train, remain=138, use_ratio=False)

    cartoon_x_public, cartoon_y_public, cartoon_x_train, cartoon_y_train = split_public_dataset(
        x=cartoon_x_train,
        y=cartoon_y_train, remain=375, use_ratio=False)

    photo_x_public, photo_y_public, photo_x_train, photo_y_train = split_public_dataset(
        x=photo_x_train,
        y=photo_y_train, remain=336, use_ratio=False)

    sketch_x_public, sketch_y_public, sketch_x_train, sketch_y_train = split_public_dataset(
        x=sketch_x_train,
        y=sketch_y_train, remain=143, use_ratio=False)

    public_X = [art_painting_x_public, cartoon_x_public, photo_x_public, sketch_x_public]
    public_Y = [art_painting_y_public, cartoon_y_public, photo_y_public, sketch_y_public]

    ################################################################################
    art_train_splits = split_dataset_iid_pacs(x=art_painting_x_train, y=art_painting_y_train, M=3, train=True)
    art_test_splits = split_dataset_iid_pacs(x=art_painting_x_test, y=art_painting_y_test, M=3, train=False)
    art_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in art_train_splits]
    art_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in art_test_splits]
    ################################################################################
    cartoon_train_splits = split_dataset_iid_pacs(x=cartoon_x_train, y=cartoon_y_train, M=3, train=True)
    cartoon_test_splits = split_dataset_iid_pacs(x=cartoon_x_test, y=cartoon_y_test, M=3, train=False)
    cartoon_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in cartoon_train_splits]
    cartoon_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in cartoon_test_splits]
    ################################################################################
    photo_train_splits = split_dataset_iid_pacs(x=photo_x_train, y=photo_y_train, M=2, train=True)
    photo_test_splits = split_dataset_iid_pacs(x=photo_x_test, y=photo_y_test, M=2, train=False)
    photo_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in photo_train_splits]
    photo_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in photo_test_splits]
    ################################################################################
    sketch_train_splits = split_dataset_iid_pacs(x=sketch_x_train, y=sketch_y_train, M=6, train=True)
    sketch_test_splits = split_dataset_iid_pacs(x=sketch_x_test, y=sketch_y_test, M=6, train=False)
    sketch_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in sketch_train_splits]
    sketch_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in sketch_test_splits]

    ##################
    #     PUBLIC     #
    ##################
    public_loader = DataLoader(
        PublicImageDataset(public_X=public_X, public_Y=public_Y),
        batch_size=hp.batchSize, shuffle=True)

    train_loaders = []
    test_loaders = []
    train_loaders += art_train_loaders
    train_loaders += cartoon_train_loaders
    train_loaders += photo_train_loaders
    train_loaders += sketch_train_loaders

    test_loaders += art_test_loaders
    test_loaders += cartoon_test_loaders
    test_loaders += photo_test_loaders
    test_loaders += sketch_test_loaders

    CIS = []
    CIS += ["art"] * len(art_train_loaders)
    CIS += ["cartoon"] * len(cartoon_train_loaders)
    CIS += ["photo"] * len(photo_train_loaders)
    CIS += ["sketch"] * len(sketch_train_loaders)

    stats = {"split": [loader.sampler.num_samples for loader in train_loaders],
             "test_split": [loader.dataset.labels.shape[0] for loader in test_loaders],
             "client_identities": CIS}

    return train_loaders, test_loaders, public_loader, stats


def plotPics(datasets, image_name='mnist', h=7, w=7, offset=100, filename="out.jpg"):
    fig, axes = plt.subplots(h, w, figsize=(h + 4, w + 4))
    for i in range(h):
        for j in range(w):
            index = offset + i * w + j + 1 + random.randint(1, 1000)  # add random number
            if image_name == 'mnist':
                data, labels = datasets
                image, label = data[index], labels[index]
                image = image.reshape(28, 28)
            elif image_name in ['mnistm', 'svhn', 'synthetic']:
                image, label = datasets[index][0], datasets[index][1]
                image = np.transpose(image, (0, 1, 2))
            else:
                assert image_name == 'usps'
                data, labels = datasets
                image, label = data[index], labels[index]
                image = image.reshape(16, 16)

            axes[i][j].set_axis_off()
            axes[i][j].imshow(image, cmap='gray', interpolation='nearest')
            # axes[i][j].set_title(f"Digit:{label}")

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0, )  # remove blank space
    fig.suptitle(f"{image_name}", x=0.5, y=1)
    plt.show()
    fig.savefig(filename)


class OFFICEHOME(Dataset):
    def __init__(self, root_path, domain, train=True, transform=None, target_transform=None):
        self.root = f"{root_path}/{domain}"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        label_name_list = os.listdir(self.root)
        self.label = []
        self.data = []

        if not os.path.exists(f"{root_path}/precessed"):
            os.makedirs(f"{root_path}/precessed")
        if os.path.exists(f"{root_path}/precessed/{domain}_data.pt") and os.path.exists(
                f"{root_path}/precessed/{domain}_label.pt"):
            print(f"Load {domain} data and label from cache.")
            self.data = torch.load(f"{root_path}/precessed/{domain}_data.pt")
            self.label = torch.load(f"{root_path}/precessed/{domain}_label.pt")
        else:
            print(f"Getting {domain} datasets")
            for index, label_name in enumerate(label_name_list):
                label_name_2_index = {'Alarm_Clock': 0, 'Backpack': 1, 'Batteries': 2, 'Bed': 3, 'Bike': 4, 'Bottle': 5,
                                      'Bucket': 6, 'Calculator': 7, 'Calendar': 8, 'Candles': 9, 'Chair': 10,
                                      'Clipboards': 11, 'Computer': 12, 'Couch': 13, 'Curtains': 14, 'Desk_Lamp': 15,
                                      'Drill': 16, 'Eraser': 17, 'Exit_Sign': 18, 'Fan': 19, 'File_Cabinet': 20,
                                      'Flipflops': 21, 'Flowers': 22, 'Folder': 23, 'Fork': 24, 'Glasses': 25,
                                      'Hammer': 26, 'Helmet': 27, 'Kettle': 28, 'Keyboard': 29, 'Knives': 30,
                                      'Lamp_Shade': 31, 'Laptop': 32, 'Marker': 33, 'Monitor': 34, 'Mop': 35,
                                      'Mouse': 36, 'Mug': 37, 'Notebook': 38, 'Oven': 39, 'Pan': 40, 'Paper_Clip': 41,
                                      'Pen': 42, 'Pencil': 43, 'Postit_Notes': 44, 'Printer': 45, 'Push_Pin': 46,
                                      'Radio': 47, 'Refrigerator': 48, 'Ruler': 49, 'Scissors': 50, 'Screwdriver': 51,
                                      'Shelf': 52, 'Sink': 53, 'Sneakers': 54, 'Soda': 55, 'Speaker': 56, 'Spoon': 57,
                                      'TV': 58, 'Table': 59, 'Telephone': 60, 'ToothBrush': 61, 'Toys': 62,
                                      'Trash_Can': 63, 'Webcam': 64}

                images_list = os.listdir(f"{self.root}/{label_name}")
                for img_name in images_list:
                    img = Image.open(f"{self.root}/{label_name}/{img_name}").convert('RGB')
                    img = np.array(img)
                    self.label.append(label_name_2_index[label_name])
                    if self.transform is not None:
                        img = self.transform(img)
                    self.data.append(img)
            self.data = torch.stack(self.data)
            self.label = torch.tensor(self.label, dtype=torch.long)
            torch.save(self.data, f"{root_path}/precessed/{domain}_data.pt")
            torch.save(self.label, f"{root_path}/precessed/{domain}_label.pt")

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_office_home(hp, verbose=False):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    art = OFFICEHOME(root_path=f"{DATA_PATH}/HOME", domain='Art', transform=transform)
    clipart = OFFICEHOME(root_path=f"{DATA_PATH}/HOME", domain='Clipart', transform=transform)
    product = OFFICEHOME(root_path=f"{DATA_PATH}/HOME", domain='Product', transform=transform)
    realworld = OFFICEHOME(root_path=f"{DATA_PATH}/HOME", domain='RealWorld', transform=transform)

    print("art", art.data.shape)
    print("clipart", clipart.data.shape)
    print("product", product.data.shape)
    print("realworld", realworld.data.shape)

    """OFFICE-HOME数据集划分
    每个domain的remain随机划分到public上面
    剩余的数据每个客户端划分成800(train):200(test)
    """
    ##################
    #     PUBLIC     #
    ##################
    art_x_public, art_y_public, art_x, art_y = split_public_dataset(art.data.numpy(), art.label.numpy(), remain=427,
                                                                    use_ratio=False)
    clipart_x_public, clipart_y_public, clipart_x, clipart_y = split_public_dataset(clipart.data.numpy(),
                                                                                    clipart.label.numpy(), remain=365,
                                                                                    use_ratio=False)
    product_x_public, product_y_public, product_x, product_y = split_public_dataset(product.data.numpy(),
                                                                                    product.label.numpy(), remain=439,
                                                                                    use_ratio=False)
    realworld_x_public, realworld_y_public, realworld_x, realworld_y = split_public_dataset(realworld.data.numpy(),
                                                                                            realworld.label.numpy(),
                                                                                            remain=357, use_ratio=False)
    public_X = [art_x_public, clipart_x_public, product_x_public, realworld_x_public]
    public_Y = [art_y_public, clipart_y_public, product_y_public, realworld_y_public]

    public_loader = DataLoader(
        PublicImageDataset(public_X=public_X, public_Y=public_Y),
        batch_size=hp.batchSize, shuffle=True)

    # TODO：训练集和测试集划分
    ##################
    #   Train&Test   #
    ##################
    test_size = 0.2
    random_state = 42
    art_x_train, art_x_test, art_y_train, art_y_test = train_test_split(art_x, art_y,
                                                                        test_size=test_size, random_state=random_state)
    clipart_x_train, clipart_x_test, clipart_y_train, clipart_y_test = train_test_split(clipart_x,
                                                                                        clipart_y,
                                                                                        test_size=test_size,
                                                                                        random_state=random_state)
    product_x_train, product_x_test, product_y_train, product_y_test = train_test_split(product_x,
                                                                                        product_y,
                                                                                        test_size=test_size,
                                                                                        random_state=random_state)
    realworld_x_train, realworld_x_test, realworld_y_train, realworld_y_test = train_test_split(realworld_x,
                                                                                                realworld_y,
                                                                                                test_size=test_size,
                                                                                                random_state=random_state)

    print(f"art_painting: # of training samples: {art_x_train.shape[0]}, # of test samples: {art_x_test.shape[0]}")
    print(
        f"clipart_painting: # of training samples: {clipart_x_train.shape[0]}, # of test samples: {clipart_x_test.shape[0]}")
    print(
        f"product_painting: # of training samples: {product_x_train.shape[0]}, # of test samples: {product_x_test.shape[0]}")
    print(
        f"realworld_painting: # of training samples: {realworld_x_train.shape[0]}, # of test samples: {realworld_x_test.shape[0]}")

    ################################################################################
    art_train_splits = split_dataset_iid_home(x=art_x_train, y=art_y_train, M=2)
    art_test_splits = split_dataset_iid_home(x=art_x_test, y=art_y_test, M=2)
    art_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in art_train_splits]
    art_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in art_test_splits]
    ################################################################################
    clipart_train_splits = split_dataset_iid_home(x=clipart_x_train, y=clipart_y_train, M=4)
    clipart_test_splits = split_dataset_iid_home(x=clipart_x_test, y=clipart_y_test, M=4)
    clipart_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in clipart_train_splits]
    clipart_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in clipart_test_splits]
    ################################################################################
    product_train_splits = split_dataset_iid_home(x=product_x_train, y=product_y_train, M=4)
    product_test_splits = split_dataset_iid_home(x=product_x_test, y=product_y_test, M=4)
    product_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in product_train_splits]
    product_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in product_test_splits]
    ################################################################################
    realworld_train_splits = split_dataset_iid_home(x=realworld_x_train, y=realworld_y_train, M=4)
    realworld_test_splits = split_dataset_iid_home(x=realworld_x_test, y=realworld_y_test, M=4)
    realworld_train_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in realworld_train_splits]
    realworld_test_loaders = [
        DataLoader(CustomImageDataset(inputs=x, labels=y, ),
                   batch_size=hp.batchSize, shuffle=True) for x, y in realworld_test_splits]
    ################################################################################

    train_loaders = []
    test_loaders = []
    train_loaders += art_train_loaders
    train_loaders += clipart_train_loaders
    train_loaders += product_train_loaders
    train_loaders += realworld_train_loaders

    test_loaders += art_test_loaders
    test_loaders += clipart_test_loaders
    test_loaders += product_test_loaders
    test_loaders += realworld_test_loaders

    CIS = []
    CIS += ["art"] * len(art_train_loaders)
    CIS += ["clipart"] * len(clipart_train_loaders)
    CIS += ["product"] * len(product_train_loaders)
    CIS += ["realworld"] * len(realworld_train_loaders)

    stats = {"split": [loader.sampler.num_samples for loader in train_loaders],
             "test_split": [loader.dataset.labels.shape[0] for loader in test_loaders],
             "client_identities": CIS}

    return train_loaders, test_loaders, public_loader, stats


def split_dataset_iid_home(x, y, M=None, verbose=False):
    """
    :param M: 客户端数量
    """
    N = y.shape[0]
    samples_per_client = int(N / M)
    clients_split = []
    ids = [i for i in range(N)]
    np.random.shuffle(ids)
    # samples_per_client = int(N * hp['ratio'] // M)
    for i in range(M):
        ids_tmp = ids[i * samples_per_client: (i + 1) * samples_per_client]
        x_tmp = x[ids_tmp]
        y_tmp = y[ids_tmp]
        clients_split += [(x_tmp, y_tmp)]

    return clients_split


if __name__ == '__main__':
    hp = {
        'dataset': 'mnist',
        'nClients': 100,
        'batchSize': 50,
        'seed': 24,
        'smallDataRatio': 0.0,
        'alpha': 0.0,
        'K': 2,
        'ratio': 1.0
    }
    # train_loaders, test_loaders, public_loader, stats = get_data_loaders(hp=hp, verbose=True)
    # # for loader in train_loaders:
    # #     for step, (x, y) in enumerate(loader):
    # #         print(x.shape)
    # #         break
    #
    # print(stats)
    #
    # users_cluster_identity = stats['users_cluster_identity']
    # print(list(users_cluster_identity))
    # print("nClients", hp['nClients'], "alpha", hp['alpha'], "nClusters", hp['K'])
    # for i in range(max(users_cluster_identity) + 1):
    #     print(np.where(np.array(users_cluster_identity) == i)[0].shape)
    # get_office(hp=hp)
    get_office_home(hp=hp)
