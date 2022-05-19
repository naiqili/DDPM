import torch
import torchvision as tv
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm.auto import tqdm, trange
import tensorflow as tf
import time


def get_flow_repr(ae_repr, flow_model, device_name='cpu', normalize=False):
    if flow_model:
        flow_model = flow_model.to(device_name)
        ae_repr = ae_repr.to(device_name)
        flow_model.eval()
        flow_repr = flow_model.f(ae_repr)[0]
    else:
        flow_repr = ae_repr

    if normalize:
        flow_repr = (flow_repr - torch.mean(flow_repr)) / torch.std(flow_repr)  #整体标准化
        flow_repr = flow_repr.to(device_name)
    return flow_repr.detach().to(device_name)


class AEDataset(Dataset):
    def __init__(self, ds_raw, ds_ae_repr, n_sample_load=-1, normalize=True, idx_load=None):
        n_sample_load = len(ds_ae_repr) if n_sample_load == -1 else n_sample_load
        self.ds_raw = ds_raw
        if idx_load is None:
            self.ae_x = ds_ae_repr[:n_sample_load]
        else:
            self.ae_x = ds_ae_repr[idx_load]

        if normalize:
            print(torch.mean(self.ae_x))
            print(torch.std(self.ae_x))
            self.ae_x = (self.ae_x - torch.mean(self.ae_x)) / torch.std(self.ae_x)
        if idx_load is None:
            self.data, self.targets = ds_raw.data[:n_sample_load], ds_raw.targets[:n_sample_load]
        else:
            self.data, self.targets = ds_raw.data[idx_load], ds_raw.targets[idx_load]

    def __getitem__(self, index):
        data, target = self.ds_raw[index]  # default transformation invoked
        return data, self.ae_x[index], target, index

    def __len__(self):
        return len(self.targets)


class OtherAEDataset(Dataset):
    def __init__(self, X, Y, Z, n_sample_load=-1, normalize=True, idx_load=None):
        # for VaDE pretrainde reuters10k, har, mnist
        
        n_sample_load = len(Y) if n_sample_load == -1 else n_sample_load
        
        if idx_load is None:
            self.ae_x = torch.tensor(Z[:n_sample_load]).float()
        else:
            self.ae_x = torch.tensor(Z[idx_load]).float()

        if normalize:
            self.ae_x = (self.ae_x - torch.mean(self.ae_x)) / torch.std(self.ae_x)
        if idx_load is None:
            self.data, self.targets = X[:n_sample_load], Y[:n_sample_load]
        else:
            self.data, self.targets = X[idx_load], Y[idx_load]

    def __getitem__(self, index):
        return self.data[index], self.ae_x[index], self.targets[index], index

    def __len__(self):
        return len(self.targets)


class InfiniteDataLoader(DataLoader):
    """
    reload the dataset from start when meets an end.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def load_dataset(name, path, split='train', download=False):
    train = (split == 'train')
    root = path
    if name == 'cifar10':
        transform = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        ds = datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=None,
                                download=download)
        ds.targets = torch.from_numpy(np.asarray(ds.targets))
        return ds
    elif name == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        ds = datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=train_transform,
                                 target_transform=None,
                                 download=download)
        ds.targets = torch.from_numpy(np.asarray(ds.targets))
        return ds
    elif name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,))  # normalize inputs
        ])
        ds = datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=None,
                              download=download)
        ds.targets = torch.from_numpy(np.asarray(ds.targets))
        return ds
    elif name == 'stl10':
        ds = datasets.STL10(root=root,
                              split=split,
                              transform=None,  # todo normalize
                              target_transform=None,
                              download=download)
        ds.targets = torch.from_numpy(np.asarray(ds.labels))
        return ds


class AEMNIST(nn.Module):
    def __init__(self, num_features, dimin=28 * 28, p=0.2):
        super(AEMNIST, self).__init__()
        self.dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(dimin, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2000)
        self.fc4 = nn.Linear(2000, num_features)
        self.relu = nn.ReLU()
        self.fc_d1 = nn.Linear(500, dimin)
        self.fc_d2 = nn.Linear(500, 500)
        self.fc_d3 = nn.Linear(2000, 500)
        self.fc_d4 = nn.Linear(num_features, 2000)
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def setP(self, p):
        self.dropout.p = p

    def setPretrain(self, mode):
        """To set training mode to pretrain or not, 
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode

    def forward1(self, x):
        x_ae = x
        x = x.view(-1, 1 * 28 * 28)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.dropout(x)
        x = self.fc_d1(x)
        x_de = x.view(-1, 1, 28, 28)
        return x_ae, x_de

    def forward2(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x_ae = x
        #
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        #
        x = self.dropout(x)
        x = self.fc_d2(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de

    def forward3(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.fc2(x)
        x = self.relu(x)
        x_ae = x
        #
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        #
        x = self.dropout(x)
        x = self.fc_d3(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de

    def forward4(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.fc2(x)
        x = self.relu(x)
        #
        x = self.fc3(x)
        x = self.relu(x)
        x_ae = x
        #
        x = self.dropout(x)
        x = self.fc4(x)
        #
        x = self.dropout(x)
        x = self.fc_d4(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        #
        x = self.fc2(x)
        x = self.relu(x)
        #
        x = self.fc3(x)
        x = self.relu(x)
        #
        x = self.fc4(x)
        x_ae = x
        # if not in pretrain mode, we only need encoder
        if self.pretrainMode == False:
            return x
        #
        x = self.fc_d4(x)
        x = self.relu(x)
        #
        x = self.fc_d3(x)
        x = self.relu(x)
        #
        x = self.fc_d2(x)
        x = self.relu(x)
        #
        x = self.fc_d1(x)
        x_de = x.view(-1, 1, 28, 28)
        return x_ae, x_de

    def decode(self, x):
        x = self.fc_d4(x)
        x = self.relu(x)
        #
        x = self.fc_d3(x)
        x = self.relu(x)
        #
        x = self.fc_d2(x)
        x = self.relu(x)
        #
        x = self.fc_d1(x)
        x_de = x.view(-1, 1, 28, 28)
        return x_de


class AEDEC(nn.Module):
    def __init__(self, input_dim):
        super(AEDEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AES2CIFAR10(nn.Module):
    def __init__(self):
        super(AES2CIFAR10, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AECIFAR10(nn.Module):
    def __init__(self):
        super(AECIFAR10, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(48, 96, 4, stride=2),  # [batch, 48, 1, 1]
            # nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2),  # [batch, 48, 4, 4]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def save_ae(epoch_id, ae_model, save_name, save_dir):
    print(f'Saving Model at Epoch {epoch_id}')
    if not os.path.exists(f'./{save_dir}'):
        os.mkdir(f'./{save_dir}')
    torch.save(ae_model, f'./{save_dir}/ae_{save_name}_E{epoch_id}.pt')


def trans_ae_data(ae_model, dataset, device_name, save_name, save_dir):
    # transfrom data
    ae_model.to(device_name)
    ae_model.eval()
    dl = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    results = []
    for i, (inputs, targets) in enumerate(dl, 0):
        inputs = inputs.to(device_name)
        z = ae_model.encoder(inputs)
        results.append(z)
    all_z = torch.cat(results, dim=0).detach().cpu()
    all_z = all_z.reshape(all_z.shape[0], -1)
    torch.save(all_z, f'./{save_dir}/ae_z_{save_name}.pt')
    return all_z


# tested only on cifar10
def train_ae(save_name, ae_model, dataset, n_epoch, save_dir, gpu_id, batch_size=64):
    print(f'training ae for {save_name}')
    tb_logger = tf.summary.create_file_writer(os.path.join(save_dir, f'ae_tb_log/{save_name}'))
    N_BATCH= np.ceil(len(dataset)/batch_size)
    device_name = 'cpu' if gpu_id == -1 else f'cuda:{gpu_id}'

    # Define an optimizer and criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ae_model = ae_model.to(device_name)
    with tb_logger.as_default():
        for epoch in trange(n_epoch):
            for i, (inputs, _) in enumerate(dl, 0):
                inputs = inputs.to(device_name)
                encoded, outputs = ae_model(inputs)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tf.summary.scalar('loss', loss.detach().cpu(), step=epoch * N_BATCH + i)
                if i % 10 == 9:
                    tb_logger.flush()
            tf.summary.scalar('epoch', epoch, step=epoch)
            if epoch == n_epoch-1 or epoch%10 == 1:
                save_ae(epoch, ae_model, save_name, save_dir)


def load_ae_ds(model_path, data_path):
    # return model, z
    return torch.load(model_path), torch.load(data_path)

if __name__ == '__main__':
    from torch.utils.data import TensorDataset
    X = np.load('../VaDE/VaDE_stl_full_X.npz')['arr_0']
    Y = np.load('../VaDE/VaDE_stl_full_Y.npz')['arr_0']
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    train_ae('ae_stl2', AEDEC(2048), ds, 500, 'saved_ae', 3)
