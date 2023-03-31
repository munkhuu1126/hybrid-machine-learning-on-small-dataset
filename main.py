import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch import nn
from common import NeuralNetwork, CNN, CIFARCNN, ResNet_18
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import os
from datetime import datetime as dt
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"


def models(choice):
    if choice.lower() == 'nn':
        return NeuralNetwork().to(device)
    elif choice.lower() == 'cnn':
        return CNN().to(device)
    elif choice.lower() == 'cifarcnn':
        return CIFARCNN().to(device)
    elif choice.lower() == 'resnet18':
        return ResNet_18(3, 100).to(device)
    else:
        assert choice == 'nn' or choice == 'cnn', f"The following model is not in the choices or haven't been made yet: {choice}"


def dataload(first_split):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409),
                             (0.2673, 0.2564, 0.2762))
    ])
    training_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=transform_train
    )
    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=transform_test
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
        loss2.backward(retain_graph=True)

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
    print(f'test loss: {test_loss}')
    return 100*correct


def main(splits, batch_sizes, model_choice):
    accuracy1 = []
    accuracy2 = []
    epochs = 10
    for batch_size in batch_sizes:
        for split in splits:
            train_data1, train_data2, test_data = dataload(split)
            train_dataloader1 = DataLoader(train_data1, batch_size=batch_size)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
            model = models(model_choice)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            for t in range(epochs):
                print(
                    f"Single Train Epoch {t+1}\n-----------------------------------------")

                train_one_data(dataloader=train_dataloader1, model=model,
                               loss_fn=loss_fn, optimizer=optimizer)
                accuracy = test(test_dataloader, model, loss_fn)
                print(
                    f"{batch_size}, {split}, single:{accuracy}")
            accuracy1.append(accuracy)
            del model
            torch.cuda.empty_cache()
    for batch_size in batch_sizes:
        for split in splits:
            train_data1, train_data2, test_data = dataload(split)
            train_dataloader1 = DataLoader(train_data1, batch_size=batch_size)
            train_dataloader2 = DataLoader(train_data2, batch_size=batch_size)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
            model2 = models(model_choice)
            loss_fn = nn.CrossEntropyLoss()
            loss_fn2 = nn.MSELoss()
            optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)
            for t in range(epochs):
                print(
                    f"Double Train Epoch {t+1}\n-----------------------------------------")
                train_both(train_dataloader1, train_dataloader2,
                           model2, loss_fn, loss_fn2, optimizer)
                accuracy = test(test_dataloader, model2, loss_fn)
            print(
                f"{batch_size}, {split},  both: {accuracy}")
            accuracy2.append(accuracy)
            del model2
            torch.cuda.empty_cache()
    d = {
        'Batch/Sizes': ['8', '32', '64', '128', 'AVERAGE'],
        '10': [accuracy1[0], accuracy1[4], accuracy1[8], accuracy1[12], sum([accuracy1[0], accuracy1[4], accuracy1[8], accuracy1[12]])/4],
        '30': [accuracy1[1], accuracy1[5], accuracy1[9], accuracy1[13],  sum([accuracy1[1], accuracy1[5], accuracy1[9], accuracy1[13]])/4],
        '50': [accuracy1[2], accuracy1[6], accuracy1[10], accuracy1[14], sum([accuracy1[2], accuracy1[6], accuracy1[10], accuracy1[14]])/4],
        '70': [accuracy1[3], accuracy1[7], accuracy1[11], accuracy1[15], sum([accuracy1[3], accuracy1[7], accuracy1[11], accuracy1[15]])/4],
        '10/90': [accuracy2[0], accuracy2[4], accuracy2[8], accuracy2[12], sum([accuracy2[0], accuracy2[4], accuracy2[8], accuracy2[12]])/4],
        '30/70': [accuracy2[1], accuracy2[5], accuracy2[9], accuracy2[13], sum([accuracy2[1], accuracy2[5], accuracy2[9], accuracy2[13]])/4],
        '50/50': [accuracy2[2], accuracy2[6], accuracy2[10], accuracy2[14], sum([accuracy2[2], accuracy2[6], accuracy2[10], accuracy2[14]])/4],
        '70/30': [accuracy2[3], accuracy2[7], accuracy2[11], accuracy2[15], sum([accuracy2[3], accuracy2[7], accuracy2[11], accuracy2[15]])/4],
    }
    df = pd.DataFrame(data=d)
    newpath = r'./results'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    df.to_csv(f'./results/{dt.now().isoformat()}.csv', sep="\t", index=False)
    print(
        f'-------------------------{model_choice} Results----------------------------------\n')
    print(f"the results have been save to /results directory")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-split', "--split_size",
                           help="Enter split portion ex=.3_.5_.7")
    argParser.add_argument('-batch', "--batch_size",
                           help="Enter batch sizes ex=8_32_64_128")
    argParser.add_argument('-model', '--model',
                           help="Model Choice [nn, cnn, yolo]")
    args = argParser.parse_args()
    splits = [float(i) for i in list(args.split_size.split('_'))]
    batch_sizes = [int(i) for i in list(args.batch_size.split('_'))]

    model = args.model
    main(splits, batch_sizes, model_choice=model)


# paste it to terminal and run
# python3 main.py -split .1_.3_.5_.7 -batch 8_32_64_128 -model nn
