import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json
import os

torch.manual_seed(42)


class CNN(torch.nn.Module):
    def __init__(self, n_hidden1, n_hidden2, out_channels1=32, out_channels2=64, dropout=0.0,
                 batch_norm=False):
        super().__init__()
        self.conv1 = torch.nn.Sequential()
        self.conv1.add_module("Conv1", torch.nn.Conv2d(in_channels=1,
                                                       out_channels=out_channels1,
                                                       kernel_size=5,
                                                       stride=1,
                                                       padding=2))
        if batch_norm and dropout == 0:
            self.conv1.add_module("BN1", torch.nn.BatchNorm2d(num_features=32))
        self.conv1.add_module("Relu1", torch.nn.ReLU())
        self.conv1.add_module("MaxPol1", torch.nn.MaxPool2d(kernel_size=2))

        self.conv2 = torch.nn.Sequential()
        self.conv2.add_module("Conv2", torch.nn.Conv2d(in_channels=out_channels1,
                                                       out_channels=out_channels2,
                                                       kernel_size=5,
                                                       stride=1,
                                                       padding=2))
        if batch_norm and dropout == 0:
            self.conv2.add_module("BN2", torch.nn.BatchNorm2d(num_features=64))
        self.conv2.add_module("Relu2", torch.nn.ReLU())
        self.conv2.add_module("MaxPol2", torch.nn.MaxPool2d(kernel_size=2))

        # fully connected layer, output 10 classes
        self.layer1 = torch.nn.Sequential()
        self.layer1.add_module("Flatten", torch.nn.Flatten())
        # 7x7 is the spatial size after two max-pooling layers
        self.layer1.add_module("Layer1", torch.nn.Linear(out_channels2 * 7 * 7, n_hidden1))
        self.layer1.add_module("Relu3", torch.nn.ReLU())
        self.layer1.add_module("Dropout1", torch.nn.Dropout(dropout))
        if batch_norm and dropout == 0:
            self.layer1.add_module("BN3", torch.nn.BatchNorm1d(num_features=n_hidden1))

        self.layer2 = torch.nn.Sequential()
        self.layer2.add_module("Layer2", torch.nn.Linear(n_hidden1, n_hidden2))
        self.layer2.add_module("Relu4", torch.nn.ReLU())
        if batch_norm and dropout == 0:
            self.layer2.add_module("BN4", torch.nn.BatchNorm1d(num_features=n_hidden2))
        self.layer2.add_module("Dropout2", torch.nn.Dropout(dropout))

        n_outputs = 10
        self.fc = torch.nn.Linear(n_hidden2, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.fc(x)
        return output


class MLP(torch.nn.Module):
    def __init__(self, n_hidden1, n_hidden2, dropout=0.0, batch_norm=False):
        super().__init__()
        n_inputs = 28 * 28
        n_outputs = 10
        self.layer1 = torch.nn.Sequential()
        self.layer1.add_module("Flatten", torch.nn.Flatten())
        self.layer1.add_module("Layer1", torch.nn.Linear(n_inputs, n_hidden1))
        self.layer1.add_module("Relu1", torch.nn.ReLU())
        self.layer1.add_module("Dropout1", torch.nn.Dropout(dropout))
        if batch_norm and dropout == 0:
            self.layer1.add_module("BN1", torch.nn.BatchNorm1d(num_features=n_hidden1))

        self.layer2 = torch.nn.Sequential()
        self.layer2.add_module("Layer2", torch.nn.Linear(n_hidden1, n_hidden2))
        self.layer2.add_module("Relu2", torch.nn.ReLU())
        if batch_norm and dropout == 0:
            self.layer2.add_module("BN2", torch.nn.BatchNorm1d(num_features=n_hidden2))
        self.layer2.add_module("Dropout2", torch.nn.Dropout(dropout))

        self.fc = torch.nn.Linear(n_hidden2, n_outputs)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x


class Softmax(torch.nn.Module):
    def __init__(self, dropout=0, batch_norm=False):
        super().__init__()
        n_inputs = 28 * 28
        n_outputs = 10
        self.linear = torch.nn.Sequential(torch.nn.Flatten(),
                                          torch.nn.Linear(n_inputs, n_outputs),
                                          torch.nn.Dropout(dropout))
        if batch_norm and dropout == 0:
            self.linear.add_module("BN", torch.nn.BatchNorm1d(num_features=n_outputs))

    def forward(self, x):
        return self.linear(x)


def run(optimizer, model):
    results = {'loss_train': [],
               'accuracy_train': [],
               'loss_test': [],
               'accuracy_test': []
               }
    loss_train = []
    loss_test = []

    criterion = torch.nn.CrossEntropyLoss()
    for _ in tqdm(range(epochs)):
        # training
        correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
        accuracy = 100 * (correct.item()) / len(train_data)
        results['accuracy_train'].append(accuracy)
        results['loss_train'].append(sum(loss_train) / len(loss_train))

        # testing
        correct = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss_test.append(loss.item())
            correct += (predicted == labels).sum()
        accuracy = 100 * (correct.item()) / len(test_data)
        results['accuracy_test'].append(accuracy)
        results['loss_test'].append(sum(loss_test) / len(loss_test))

    return results


def run_softmax(d, bn):
    test = 'softmax_dropout_{}_bnorm_{}'.format(d, bn)
    print(test)
    model_softmax = Softmax(dropout=d, batch_norm=bn)
    optimizer_softmax = torch.optim.SGD(model_softmax.parameters(), lr=lr)
    results = run(optimizer_softmax, model_softmax)
    save_result(test, results)


def run_mlp(h1, h2, d, bn):
    test = 'mlp_h1_{}_h2_{}_dropout_{}_bnorm_{}'.format(h1, h2, d, bn)
    print(test)
    model_mlp = MLP(n_hidden1=h1, n_hidden2=h2, dropout=d, batch_norm=bn)
    optimizer_mlp = torch.optim.SGD(model_mlp.parameters(), lr=lr)
    results = run(optimizer_mlp, model_mlp)
    save_result(test, results)


def run_cnn(h1, h2, d, bn, oc1, oc2):
    test = 'cnn_h1_{}_h2_{}_dropout_{}_bnorm_{}_oc1_{}_oc2_{}'.format(h1, h2, d, bn, oc1, oc2)
    print(test)
    model_cnn = CNN(n_hidden1=h1, n_hidden2=h2, dropout=d, batch_norm=bn, out_channels1=oc1, out_channels2=oc2)
    optimizer_cnn = torch.optim.SGD(model_cnn.parameters(), lr=lr)
    results = run(optimizer_cnn, model_cnn)
    save_result(test, results)


def save_result(filename, results):
    json_object = json.dumps(results)
    path = 'output/{}.json'.format(filename)
    with open(path, "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    os.makedirs('./output', exist_ok=True)
    lr = 0.01
    epochs = 1000
    data_len = 3000
    data_batch_size = 200

    # download and apply the transform
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_data = list(train_data)[:data_len]
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    test_data = list(test_data)[:data_len]
    train_loader = DataLoader(dataset=train_data, batch_size=data_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=data_batch_size, shuffle=True)

    # Softmax: dropout, batch_norm
    run_softmax(0, False)
    run_softmax(0.2, False)
    run_softmax(0, True)

    # MLP: n_hidden1, n_hidden2, dropout, batch_norm
    run_mlp(512, 256, 0, False)
    run_mlp(512, 256, 0.2, False)
    run_mlp(512, 256, 0, True)
    run_mlp(64, 32, 0, False)
    run_mlp(1024, 512, 0, False)

    # CNN: hidden1, hidden2, dropout, batch_norm, out_channel1, out_channel2
    run_cnn(512, 256, 0, False, 32, 64)
    run_cnn(512, 256, 0.2, False, 32, 64)
    run_cnn(512, 256, 0, True, 32, 64)
    run_cnn(64, 32, 0, False, 32, 64)
    run_cnn(512, 256, 0, False, 4, 8)
    run_cnn(64, 32, 0, False, 4, 8)
