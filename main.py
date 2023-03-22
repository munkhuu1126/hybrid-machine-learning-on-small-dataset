import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from common import NeuralNetwork, CNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

def models(choice):
    if choice.lower() == 'nn':
        return NeuralNetwork().to(device)
    elif choice.lower() == 'cnn':
        return CNN().to(device)
    else: 
        assert choice == 'nn' or choice == 'cnn', f"The following model is not in the choices or haven't been made yet: {choice}"

def dataload(first_split):
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    # train_data1, train_data2 = torch.utils.data.random_split(
    #     training_data, [int(len(training_data)*first_split),
    #                     int(len(training_data)*second_split)]
    # )
    train_data1 = torch.utils.data.Subset(
        training_data, range(int(len(training_data)*first_split)))

    train_data2 = torch.utils.data.Subset(training_data, range(
        int(len(training_data)*first_split), int(len(training_data))))
    # print(len(train_data1), len(train_data2))
    return train_data1, train_data2, test_data


def train_one_data(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_both(dataloader1, dataloader2, model, loss_fn, loss_fn2, optimizer):
    size1 = len(dataloader1.dataset)
    size2 = len(dataloader2.dataset)
    model.train()
    for batch, ((X, y), (X1, y1)) in enumerate(zip(dataloader1, dataloader2)):
        X, y = X.to(device), y.to(device)
        X1 = X1.to(device)
        # prediction 1
        pred2 = model(X1)
        pred1 = model(X)

        # with label
        loss = loss_fn(pred1, y)
        # Backpropagation 1
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        pred3 = model(X1)
        # prediction 2
        loss2 = loss_fn2(pred2, pred3)
        # loss3 = loss+loss2
        # Backpropagation 2
        loss2.backward(retain_graph=False)

        optimizer.step()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch+1)*len(X)
        #     print('for X')
        #     print(f"loss: {loss:>7f} [{current:>5d}/{size1:>5d}]")
        # if batch % 100 == 0:
        #     loss2, current = loss2.item(), (batch+1)*len(X1)
        # print('for X1')
        # print(f"loss: {loss2:>7f} [{current:>5d}/{size2:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return 100*correct


def main(splits, batch_sizes, model_choice):
    for batch_size in batch_sizes:
        for split in splits:
            accuracy1 = []
            accuracy2 = []
            # first_split = split
            # second_split = 1-split
            train_data1, train_data2, test_data = dataload(split)
            train_dataloader1 = DataLoader(train_data1, batch_size=batch_size)
            train_dataloader2 = DataLoader(train_data2, batch_size=batch_size)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
            model = models(model_choice)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            epochs = 10
            for t in range(epochs):
                print(
                    f"Single Train Epoch {t+1}\n-----------------------------------------")

                train_one_data(dataloader=train_dataloader2, model=model,
                               loss_fn=loss_fn, optimizer=optimizer)
                accuracy1.append(test(test_dataloader, model, loss_fn))
            del model
            torch.cuda.empty_cache()
            model2 = models(model_choice)
            loss_fn2 = nn.MSELoss()
            optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)
            for t in range(epochs):
                print(
                    f"Double Train Epoch {t+1}\n-----------------------------------------")
                train_both(train_dataloader1, train_dataloader2,
                           model2, loss_fn, loss_fn2, optimizer)
                accuracy2.append(test(test_dataloader, model2, loss_fn))
            # plt.plot(accuracy2)
            # plt.xlabel('no of epochs')
            # plt.ylabel("accuracy")
            # plt.title("Epoch vs Accuracy")
            # plt.show()
            print(
                f"{batch_size}, {split}, loader1:{np.max(accuracy1)}, loader2: {np.max(accuracy2)}")
            del model2
            torch.cuda.empty_cache()


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-split', "--split_size",
                           help="Enter split portion ex=.3_.5_.7")
    argParser.add_argument('-batch', "--batch_size", help="Enter batch sizes ex=8_32_64_128")
    argParser.add_argument('-model', '--model',
                           help="Model Choice [nn, cnn, yolo]")
    args = argParser.parse_args()
    splits = [float(i) for i in list(args.split_size.split('_'))]
    batch_sizes = [int(i) for i in list(args.batch_size.split('_'))]
    model = args.model
    main(splits, batch_sizes, model_choice = model)

# ! default option
# python3 main.py -split .3_.5_.7 -batch 8_32_64_128 -model nn