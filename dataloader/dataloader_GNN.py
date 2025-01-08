import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, configs, args):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        print(f'This dataset has {max(y_train)+1} classes')

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.long()

        self.len = X_train.shape[0]
        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0],shape[1],configs.time_denpen_len, configs.window_size)
        self.x_data = torch.transpose(self.x_data, 1,2)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
class Load_Training_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, configs, args):
        super(Load_Training_Dataset, self).__init__()
        self.args= args
        self.configs = configs
        X_train = dataset["samples"]
        y_train = dataset["labels"]

        X_train_1 = X_train
        X_train_2 = X_train

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.x_data_1 = torch.from_numpy(X_train_1)
            self.x_data_2 = torch.from_numpy(X_train_2)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.x_data_1 = X_train_1.float()
            self.x_data_2 = X_train_2.float()
            self.y_data = y_train.long()

        self.len = X_train.shape[0]
        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0], shape[1], configs.time_denpen_len, configs.window_size)
        self.x_data = torch.transpose(self.x_data, 1, 2)

        self.x_data_aug1 = self.x_data_1.reshape(shape[0], shape[1], configs.time_denpen_len, configs.window_size)
        self.x_data_aug1 = torch.transpose(self.x_data_aug1, 1, 2)

        self.x_data_aug2 = self.x_data_2.reshape(shape[0], shape[1], configs.time_denpen_len, configs.window_size)
        self.x_data_aug2 = torch.transpose(self.x_data_aug2, 1, 2)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index],self.x_data_aug1[index],self.x_data_aug2[index]

    def __len__(self):
        return self.len

def data_generator(data_path, configs, args):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Training_Dataset(train_dataset, configs, args)
    test_dataset = Load_Dataset(test_dataset, configs, args)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size_test,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader


