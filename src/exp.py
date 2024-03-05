import torch
from modules import TscVAE, EscVAE, Linker
from funcs import t_elbo_loss, e_elbo_loss
from dataset import VaeDataManager, ZDataManager
import numpy as np

class TscVaeExperiment:
  def __init__(self, model_params, lr, weight_decay, x, xcell_id, xcell_name, test_ratio, batch_size, validation_ratio):
        self.vdm = VaeDataManager(x, xcell_id, xcell_name test_ratio, batch_size, validation_ratio)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TscVAE(**model_params)
        self.model_params = model_params
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr

    def train_epoch(self):
        self.model.train()
        for x, xcell_id, xcell_name in self.vdm.train_loader:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model.t_elbo_loss(qz, ld, x)
            loss.backward()
            self.optimizer.step()
        return(0)

    def evaluate(self):
        self.model.eval()
        x = self.vdm.validation_x.to(self.device)
        loss = self.model.t_elbo_loss(qz, ld, x)
        return(loss)

    def test(self):
        self.model.eval()
        x = self.vdm.test_x.to(self.device)
        loss = self.model.elbo_loss(qz, ld, x)
        return(loss)

    def train_total(self, epoch_num, patience):
        for epoch in range(epoch_num):
            loss = self.train_epoch()
            val_loss = self.evaluate()
            if epoch % 10 == 0:
              print(f'loss at epoch {epoch} is {val_loss}')

    def init_optimizer(self, lr, weight_decay):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)


