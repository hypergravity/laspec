import torch
import numpy as np
import datetime
from laspec.slamplus import SlamPredictor
import matplotlib.pyplot as plt
from copy import deepcopy


class NNModel(torch.nn.Module):
    def __init__(self, n_label=3, n_pixel=1900, n_hidden=300, n_layer=3, drop_rate=0, wave=None):
        super(NNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(n_label, n_hidden))
        self.layers.append(torch.nn.LeakyReLU())
        if drop_rate > 0:
            self.layers.append(torch.nn.Dropout(p=drop_rate))
        for i in range(n_layer - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))
            self.layers.append(torch.nn.LeakyReLU())
            if drop_rate > 0:
                self.layers.append(torch.nn.Dropout(p=drop_rate))
        self.layers.append(torch.nn.Linear(n_hidden, n_pixel))

        self.xmin = 0
        self.xmax = 0
        self.history = None
        self.state_dict_best = None
        self.loss_best = np.inf
        self.wave = wave

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return x

    def make_dataset(self, x, y, f_train=.9, batch_size=100, device=None):
        # normalization
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.xmin = np.min(x, axis=0)
        self.xmax = np.max(x, axis=0)
        x = (x - self.xmin.reshape(1, -1)) / (self.xmax.reshape(1, -1) - self.xmin.reshape(1, -1)) - 0.5
        # to tensor

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if device is not None:
            x = x.cuda(device=device)
            y = y.cuda(device=device)

        # split
        n_sample = x.size(0)
        n_train = np.int32(f_train * n_sample)
        # make dataset
        ds = torch.utils.data.TensorDataset(x, y)
        # train valid random split
        ds_train, ds_test = torch.utils.data.random_split(ds, [n_train, n_sample - n_train])
        # make dataloader for training set
        if True:
            # if weights even
            dl_train = torch.utils.data.DataLoader(ds_train, sampler=None, batch_size=batch_size, shuffle=True)
            dl_test = torch.utils.data.DataLoader(ds_test, sampler=None, batch_size=batch_size, shuffle=True)
        else:
            # if weights not even
            raise NotImplementedError()
            # wrs = torch.utils.data.WeightedRandomSampler([0,1,2], 10, replacement=True, generator=None)
            # dl = torch.utils.data.DataLoader(ds_train.dataset, sampler=wrs, batch_size=batch_size, shuffle=True, num_workers=1)
        return dl_train, dl_test

    def fit(self, x, y, f_train=0.1, batch_size=100, device=None, lr=1e-4, n_epoch=20000, gain_loss=1e4,
            step_verbose=10, sd_init=None, clean_history=True, restore_best=False):
        """

        Parameters
        ----------
        x:
            training labels
        y:
            training flux
        f_train:
            fraction for training, defaults to 0.9
        batch_size:
            batch size
        device:
            if you use GPUs, set device to 0, 1, etc.
            if you use CPU, set to None.
        lr:
            learning rate
        n_epoch:
            number of epochs
        gain_loss:
            defaults to 1e4
        step_verbose:
            validate model every *step_verbose* epochs
        sd_init:
            if not None, set state_dict to the sd_init
        clean_history:
            if True, clean history
        restore_best:
            if True, restore model with lowest loss

        Returns
        -------
        None
        """
        # make datasets
        dl_train, dl_test = self.make_dataset(x, y, f_train=f_train, batch_size=batch_size, device=device)

        # specify device
        if device is not None and torch.cuda.is_available():
            self.cuda(device)

        # load state dict if necessary
        if sd_init is not None:
            self.load_state_dict(sd_init)
        if clean_history:
            self.history = dict(
                epoch=[],
                loss_train_batch=[],
                loss_train=[],
                loss_test=[],
            )

        # initialization
        self.loss_best = np.inf
        self.state_dict_best = None

        # payne = Payne_model(n_layer=3,dropout=True,p=0.1)
        # payne = Payne_model(n_layer=3, dropout=False, p=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # optimize all cnn parameters
        # optimizer = torch.optim.SGD(payne.parameters(), lr=LR, momentum=.9)   # optimize all cnn parameters
        # optimizer = RAdam(payne.parameters(), lr=LR)  # optimize all cnn parameters
        loss_func = torch.nn.L1Loss(reduction="mean")

        # train model
        for epoch in range(n_epoch):
            # train mode
            self.train()
            loss_train_batch = []
            for batch_idx, (batch_x, batch_y) in enumerate(dl_train):
                optimizer.zero_grad()
                batch_y_pred = self(batch_x)
                batch_loss = loss_func(batch_y_pred, batch_y) * gain_loss
                batch_loss.backward()
                optimizer.step()
                loss_train_batch.append(batch_loss.item())
                # print("epoch {} batch {} loss={:.8f}".format(epoch, batch_idx, batch_loss.item()))
            loss_train_batch = np.mean(loss_train_batch)

            if np.mod(epoch, step_verbose) == 0:
                # test mode
                self.eval()
                with torch.no_grad():
                    loss_train = []
                    for batch_idx, (batch_x, batch_y) in enumerate(dl_train):
                        loss_train.append(loss_func(self(batch_x), batch_y).item() * gain_loss)
                    loss_train = np.mean(loss_train)

                    loss_test = []
                    for batch_idx, (batch_x, batch_y) in enumerate(dl_test):
                        loss_test.append(loss_func(self(batch_x), batch_y).item() * gain_loss)
                    loss_test = np.mean(loss_test)

                print("{} Epoch-[{:05d}/{:05d}] loss_train_batch={:.8f} loss_train={:.8f} loss_test={:.8f}".format(
                    datetime.datetime.now(), epoch, n_epoch, loss_train_batch, loss_train, loss_test))
                self.history["epoch"].append(epoch)
                self.history["loss_train_batch"].append(loss_train_batch)
                self.history["loss_train"].append(loss_train)
                self.history["loss_test"].append(loss_test)
                if loss_test < self.loss_best:
                    self.loss_best = loss_test
                    self.state_dict_best = deepcopy(self.state_dict())
        print("===========================================")
        print("best loss_test: ", self.loss_best)
        print("===========================================")
        if restore_best:
            print("Restoring the best coefficients ... \n")
            self.load_state_dict(self.state_dict_best)
        return

    def predict_spectrum(self, x_test):
        """ predict a spectrum """
        x_test = (x_test - self.xmin) / (self.xmax - self.xmin) - 0.5
        return self(x_test).detach().numpy()

    def to_sp(self):
        """ transform to SlamPredictor instance """
        w = [self.state_dict_best[k].cpu().numpy() for k in self.state_dict_best.keys() if "weight" in k]
        b = [self.state_dict_best[k].cpu().numpy()[:, None] for k in self.state_dict_best.keys() if "bias" in k]
        alpha = 0.01
        xmin = self.xmin
        xmax = self.xmax
        return SlamPredictor(w, b, alpha, xmin, xmax, wave=self.wave)

    def plot_history(self):
        """ plot training history """
        if self.history is not None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            for k in self.history.keys():
                if k is not "epoch":
                    ax.plot(self.history["epoch"], self.history[k], lw=2, label=k)
            ax.legend(fontsize=15)
            ax.set_ylim(0, 2*np.median(self.history["loss_test"]))
            ax.set_xlabel("Epoch", fontsize=15)
            fig.show()
        else:
            print("@NNModel: the model is not trained yet!")
