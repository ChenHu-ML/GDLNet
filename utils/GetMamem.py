import os
import torch
import torch.utils.data as data
from scipy import io


def getAllDataloader(subject, data_path='./data/MAMEM/', bs=64):
    dev = torch.device("cpu")
    train = io.loadmat(os.path.join(data_path, 'U' + f'{subject:03d}' + '.mat'))
    tempdata = torch.Tensor(train['x_test']).unsqueeze(1)
    templabel = torch.Tensor(train['y_test']).view(-1)

    x_train = tempdata[:300]
    y_train = templabel[:300]
    x_valid = tempdata[300:400]
    y_valid = templabel[300:400]
    x_test = tempdata[400:500]
    y_test = templabel[400:500]

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)

    train_dataset = data.TensorDataset(x_train, y_train)
    valid_dataset = data.TensorDataset(x_valid, y_valid)
    test_dataset = data.TensorDataset(x_test, y_test)

    trainloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    validloader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    testloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return trainloader, validloader, testloader
