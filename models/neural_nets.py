import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Digit5CNNModel(nn.Module):

    def __init__(self):
        super(Digit5CNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 32, kernel_size=5))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(32))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv2', nn.Conv2d(32, 50, kernel_size=5))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop1', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))

        self.specific_feature = nn.Sequential()
        self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 32, kernel_size=5))
        self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(32))
        self.specific_feature.add_module('s_pool1', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu1', nn.ReLU(True))

        self.specific_feature.add_module('s_conv2', nn.Conv2d(32, 50, kernel_size=5))
        self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop1', nn.Dropout2d())
        self.specific_feature.add_module('s_pool2', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu2', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 4 * 4, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 10))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        invariant_feature = self.invariant_feature(input_data)
        specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 4 * 4)
        specific_feature_flatten = specific_feature.view(-1, 50 * 4 * 4)
        # concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        additive_feature = invariant_feature_flatten + specific_feature_flatten
        label_output = self.label_classifier(additive_feature)
        reverse_invariant_feature = ReverseLayerF.apply(invariant_feature_flatten, alpha)
        domain_output = self.domain_classifier(reverse_invariant_feature)

        return label_output, domain_output, invariant_feature_flatten, specific_feature_flatten


class SimpleDigit5CNNModel(nn.Module):

    def __init__(self):
        super(SimpleDigit5CNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(64))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop1', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))

        # self.specific_feature = nn.Sequential()
        # self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(64))
        # self.specific_feature.add_module('s_pool1', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu1', nn.ReLU(True))
        #
        # self.specific_feature.add_module('s_conv2', nn.Conv2d(64, 50, kernel_size=5))
        # self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        # self.specific_feature.add_module('s_drop1', nn.Dropout2d())
        # self.specific_feature.add_module('s_pool2', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu2', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 4 * 4, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 10))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        invariant_feature = self.invariant_feature(input_data)
        # specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 4 * 4)
        # specific_feature_flatten = specific_feature.view(-1, 50 * 4 * 4)
        # concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        label_output = self.label_classifier(invariant_feature_flatten)
        return label_output


class Digit5_MINE(nn.Module):
    def __init__(self):
        super(Digit5_MINE, self).__init__()
        self.fc1_x = nn.Linear(50 * 4 * 4, 512)
        self.fc1_y = nn.Linear(50 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class Office_MINE(nn.Module):
    def __init__(self):
        super(Office_MINE, self).__init__()
        self.fc1_x = nn.Linear(50 * 12 * 12, 512)
        self.fc1_y = nn.Linear(50 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class OfficeCNNModel(nn.Module):

    def __init__(self):
        super(OfficeCNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(64))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv2', nn.Conv2d(64, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop2', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn3', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop3', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool3', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu3', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv4', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn4', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop4', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool4', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu4', nn.ReLU(True))

        self.specific_feature = nn.Sequential()
        self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(64))
        self.specific_feature.add_module('s_pool1', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu1', nn.ReLU(True))

        self.specific_feature.add_module('s_conv2', nn.Conv2d(64, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop2', nn.Dropout2d())
        self.specific_feature.add_module('s_pool2', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu2', nn.ReLU(True))

        self.specific_feature.add_module('s_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn3', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop3', nn.Dropout2d())
        self.specific_feature.add_module('s_pool3', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu3', nn.ReLU(True))

        self.specific_feature.add_module('s_conv4', nn.Conv2d(50, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn4', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop4', nn.Dropout2d())
        self.specific_feature.add_module('s_pool4', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu4', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 12 * 12 * 2, 500))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(500))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(500, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 10))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 12 * 12, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        invariant_feature = self.invariant_feature(input_data)
        specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 12 * 12)
        specific_feature_flatten = specific_feature.view(-1, 50 * 12 * 12)
        concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        # additive_feature = invariant_feature_flatten + specific_feature_flatten
        label_output = self.label_classifier(concat_feature)
        reverse_invariant_feature = ReverseLayerF.apply(invariant_feature_flatten, alpha)
        domain_output = self.domain_classifier(reverse_invariant_feature)

        return label_output, domain_output, invariant_feature_flatten, specific_feature_flatten


class SimpleOfficeCNNModel(nn.Module):
    def __init__(self):
        super(SimpleOfficeCNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('i_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('i_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('i_pool1', nn.MaxPool2d(2))
        self.feature.add_module('i_relu1', nn.ReLU(True))

        self.feature.add_module('i_conv2', nn.Conv2d(64, 50, kernel_size=3))
        self.feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('i_drop2', nn.Dropout2d())
        self.feature.add_module('i_pool2', nn.MaxPool2d(2))
        self.feature.add_module('i_relu2', nn.ReLU(True))

        self.feature.add_module('i_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.feature.add_module('i_bn3', nn.BatchNorm2d(50))
        self.feature.add_module('i_drop3', nn.Dropout2d())
        self.feature.add_module('i_pool3', nn.MaxPool2d(2))
        self.feature.add_module('i_relu3', nn.ReLU(True))

        self.feature.add_module('i_conv4', nn.Conv2d(50, 50, kernel_size=3))
        self.feature.add_module('i_bn4', nn.BatchNorm2d(50))
        self.feature.add_module('i_drop4', nn.Dropout2d())
        self.feature.add_module('i_pool4', nn.MaxPool2d(2))
        self.feature.add_module('i_relu4', nn.ReLU(True))

        self.classifier = nn.Sequential()
        self.classifier = nn.Sequential()
        self.classifier.add_module('l_fc1', nn.Linear(50 * 12 * 12, 500))
        self.classifier.add_module('l_bn1', nn.BatchNorm1d(500))
        self.classifier.add_module('l_relu1', nn.ReLU(True))
        self.classifier.add_module('l_drop1', nn.Dropout())
        self.classifier.add_module('l_fc2', nn.Linear(500, 100))
        self.classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.classifier.add_module('l_relu2', nn.ReLU(True))
        self.classifier.add_module('l_fc3', nn.Linear(100, 10))
        self.classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        feature = self.feature(input_data)
        feature_flatten = feature.view(-1, 50 * 12 * 12)
        label_output = self.classifier(feature_flatten)

        return label_output


class PACSCNNModel(nn.Module):
    """BatchNorm必须要使用，否则在很多数据集上效果会很差"""

    def __init__(self):
        super(PACSCNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 32, kernel_size=11))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(32))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(4))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))
        self.invariant_feature.add_module('i_conv2', nn.Conv2d(32, 50, kernel_size=11))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop1', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(4))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))
        self.invariant_feature.add_module('i_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn3', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_pool3', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu3', nn.ReLU(True))

        self.specific_feature = nn.Sequential()
        self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 32, kernel_size=11))
        self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(32))
        self.specific_feature.add_module('s_pool1', nn.MaxPool2d(4))
        self.specific_feature.add_module('s_relu1', nn.ReLU(True))
        self.specific_feature.add_module('s_conv2', nn.Conv2d(32, 50, kernel_size=11))
        self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop1', nn.Dropout2d())
        self.specific_feature.add_module('s_pool2', nn.MaxPool2d(4))
        self.specific_feature.add_module('s_relu2', nn.ReLU(True))
        self.specific_feature.add_module('s_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn3', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_pool3', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu3', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 4 * 4, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 7))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 227, 227)
        invariant_feature = self.invariant_feature(input_data)
        specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 4 * 4)
        specific_feature_flatten = specific_feature.view(-1, 50 * 4 * 4)
        # concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        additive_feature = invariant_feature_flatten + specific_feature_flatten
        label_output = self.label_classifier(additive_feature)
        reverse_invariant_feature = ReverseLayerF.apply(invariant_feature_flatten, alpha)
        domain_output = self.domain_classifier(reverse_invariant_feature)

        return label_output, domain_output, invariant_feature_flatten, specific_feature_flatten


class SimplePACSCNNModel(nn.Module):
    """BatchNorm必须要使用，否则在很多数据集上效果会很差"""

    def __init__(self):
        super(SimplePACSCNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 64, kernel_size=11))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(64))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(4))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))
        self.invariant_feature.add_module('i_conv2', nn.Conv2d(64, 50, kernel_size=11))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop1', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(4))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))
        self.invariant_feature.add_module('i_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn3', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_pool3', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu3', nn.ReLU(True))

        # self.specific_feature = nn.Sequential()
        # self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 64, kernel_size=11))
        # self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(64))
        # self.specific_feature.add_module('s_pool1', nn.MaxPool2d(4))
        # self.specific_feature.add_module('s_relu1', nn.ReLU(True))
        # self.specific_feature.add_module('s_conv2', nn.Conv2d(64, 50, kernel_size=11))
        # self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        # self.specific_feature.add_module('s_drop1', nn.Dropout2d())
        # self.specific_feature.add_module('s_pool2', nn.MaxPool2d(4))
        # self.specific_feature.add_module('s_relu2', nn.ReLU(True))
        # self.specific_feature.add_module('s_conv3', nn.Conv2d(50, 50, kernel_size=3))
        # self.specific_feature.add_module('s_bn3', nn.BatchNorm2d(50))
        # self.specific_feature.add_module('s_pool3', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu3', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 4 * 4, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 7))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 227, 227)
        invariant_feature = self.invariant_feature(input_data)
        # specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 4 * 4)
        # specific_feature_flatten = specific_feature.view(-1, 50 * 4 * 4)
        # concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        # additive_feature = invariant_feature_flatten + specific_feature_flatten
        label_output = self.label_classifier(invariant_feature_flatten)
        return label_output


class PACS_MINE(nn.Module):
    def __init__(self):
        super(PACS_MINE, self).__init__()
        self.fc1_x = nn.Linear(50 * 4 * 4, 512)
        self.fc1_y = nn.Linear(50 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class HomeCNNModel(nn.Module):

    def __init__(self):
        super(HomeCNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 32, kernel_size=5))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(32))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv2', nn.Conv2d(32, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop2', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn3', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop3', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool3', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu3', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv4', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn4', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop4', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool4', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu4', nn.ReLU(True))

        self.specific_feature = nn.Sequential()
        self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 32, kernel_size=5))
        self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(32))
        self.specific_feature.add_module('s_pool1', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu1', nn.ReLU(True))

        self.specific_feature.add_module('s_conv2', nn.Conv2d(32, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop2', nn.Dropout2d())
        self.specific_feature.add_module('s_pool2', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu2', nn.ReLU(True))

        self.specific_feature.add_module('s_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn3', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop3', nn.Dropout2d())
        self.specific_feature.add_module('s_pool3', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu3', nn.ReLU(True))

        self.specific_feature.add_module('s_conv4', nn.Conv2d(50, 50, kernel_size=3))
        self.specific_feature.add_module('s_bn4', nn.BatchNorm2d(50))
        self.specific_feature.add_module('s_drop4', nn.Dropout2d())
        self.specific_feature.add_module('s_pool4', nn.MaxPool2d(2))
        self.specific_feature.add_module('s_relu4', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 12 * 12, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 65))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 12 * 12, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        invariant_feature = self.invariant_feature(input_data)
        specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 12 * 12)
        specific_feature_flatten = specific_feature.view(-1, 50 * 12 * 12)
        # concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        additive_feature = invariant_feature_flatten + specific_feature_flatten
        label_output = self.label_classifier(additive_feature)
        reverse_invariant_feature = ReverseLayerF.apply(invariant_feature_flatten, alpha)
        domain_output = self.domain_classifier(reverse_invariant_feature)

        return label_output, domain_output, invariant_feature_flatten, specific_feature_flatten


class SimpleHomeCNNModel(nn.Module):

    def __init__(self):
        super(SimpleHomeCNNModel, self).__init__()
        self.invariant_feature = nn.Sequential()
        self.invariant_feature.add_module('i_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.invariant_feature.add_module('i_bn1', nn.BatchNorm2d(64))
        self.invariant_feature.add_module('i_pool1', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu1', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv2', nn.Conv2d(64, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn2', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop2', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool2', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu2', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv3', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn3', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop3', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool3', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu3', nn.ReLU(True))

        self.invariant_feature.add_module('i_conv4', nn.Conv2d(50, 50, kernel_size=3))
        self.invariant_feature.add_module('i_bn4', nn.BatchNorm2d(50))
        self.invariant_feature.add_module('i_drop4', nn.Dropout2d())
        self.invariant_feature.add_module('i_pool4', nn.MaxPool2d(2))
        self.invariant_feature.add_module('i_relu4', nn.ReLU(True))

        # self.specific_feature = nn.Sequential()
        # self.specific_feature.add_module('s_conv1', nn.Conv2d(3, 64, kernel_size=5))
        # self.specific_feature.add_module('s_bn1', nn.BatchNorm2d(64))
        # self.specific_feature.add_module('s_pool1', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu1', nn.ReLU(True))
        #
        # self.specific_feature.add_module('s_conv2', nn.Conv2d(64, 50, kernel_size=3))
        # self.specific_feature.add_module('s_bn2', nn.BatchNorm2d(50))
        # self.specific_feature.add_module('s_drop2', nn.Dropout2d())
        # self.specific_feature.add_module('s_pool2', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu2', nn.ReLU(True))
        #
        # self.specific_feature.add_module('s_conv3', nn.Conv2d(50, 50, kernel_size=3))
        # self.specific_feature.add_module('s_bn3', nn.BatchNorm2d(50))
        # self.specific_feature.add_module('s_drop3', nn.Dropout2d())
        # self.specific_feature.add_module('s_pool3', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu3', nn.ReLU(True))
        #
        # self.specific_feature.add_module('s_conv4', nn.Conv2d(50, 50, kernel_size=3))
        # self.specific_feature.add_module('s_bn4', nn.BatchNorm2d(50))
        # self.specific_feature.add_module('s_drop4', nn.Dropout2d())
        # self.specific_feature.add_module('s_pool4', nn.MaxPool2d(2))
        # self.specific_feature.add_module('s_relu4', nn.ReLU(True))

        self.label_classifier = nn.Sequential()
        self.label_classifier.add_module('l_fc1', nn.Linear(50 * 12 * 12, 100))
        self.label_classifier.add_module('l_bn1', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu1', nn.ReLU(True))
        self.label_classifier.add_module('l_drop1', nn.Dropout())
        self.label_classifier.add_module('l_fc2', nn.Linear(100, 100))
        self.label_classifier.add_module('l_bn2', nn.BatchNorm1d(100))
        self.label_classifier.add_module('l_relu2', nn.ReLU(True))
        self.label_classifier.add_module('l_fc3', nn.Linear(100, 65))
        self.label_classifier.add_module('l_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha=0.1):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        invariant_feature = self.invariant_feature(input_data)
        # specific_feature = self.specific_feature(input_data)

        invariant_feature_flatten = invariant_feature.view(-1, 50 * 12 * 12)
        # specific_feature_flatten = specific_feature.view(-1, 50 * 12 * 12)
        # concat_feature = torch.cat((invariant_feature_flatten, specific_feature_flatten), dim=1)
        # additive_feature = invariant_feature_flatten + specific_feature_flatten
        label_output = self.label_classifier(invariant_feature_flatten)
        return label_output


class Home_MINE(nn.Module):
    def __init__(self):
        super(Home_MINE, self).__init__()
        self.fc1_x = nn.Linear(50 * 12 * 12, 512)
        self.fc1_y = nn.Linear(50 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    model = PACSCNNModel().cuda()
    # summary(model, input_size=(3, 28, 28))
    # summary(model, input_size=(3, 224, 224))
    summary(model, input_size=(3, 227, 227))
    # _x = torch.rand((50, 3, 224, 224)).cuda()
    # _x = torch.rand((50, 3, 28, 28)).cuda()
    _x = torch.rand((50, 3, 227, 227)).cuda()
    l_output, d_output, _, _ = model(_x)
    # l_output = model(_x)
    print(f'{_x.shape}->label output: {l_output.shape}')
    print(f'{_x.shape}->domain output: {d_output.shape}')
    classification_parameters = 0
    total_parameters = 0
    for k, v in model.state_dict().items():
        if k.startswith('invariant') or k.startswith('specific') or k.startswith('label'):
            print(k, v.numel())
            classification_parameters += v.numel()
        total_parameters += v.numel()
    print(f"Classification parameters: {classification_parameters}")
    print(f"Parameters in total {total_parameters}")
