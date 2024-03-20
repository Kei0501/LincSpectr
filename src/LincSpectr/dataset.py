import torch
import numpy as np
import torch.distributions as dist
from torchvision import transforms
from PIL import Image

class VaeDataSet(torch.utils.data.Dataset):
    def __init__(self,x,xcell_id,set_timeax=128,set_freqax=128,transform=None,pre_transform=None):
        self.x = x
        self.xcell_id = xcell_id
        self.xtime = set_timeax
        self.xfreq = set_freqax
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return(self.x.shape[0])
      
    def __getitem__(self,idx):
        idx_x = self.x[idx]
        idx_xcell_id = np.load(self.xcell_id[idx])
        set_timeax = self.xtime
        set_freqax = self.xfreq
        idx_xcell_id = np.array(Image.fromarray(idx_xcell_id).resize((set_timeax,set_freqax)))
        idx_xcell_id = self.transform(idx_xcell_id)
        idx_xcell_name = self.xcell_id[idx]
        return(idx_x,idx_xcell_id,idx_xcell_name)


class ZDataSet(torch.utils.data.Dataset):
    def __init__(self,x,xcell_id,t_vae, e_vae,set_timeax=128,set_freqax=128,transform=None,pre_transform=None):
        self.x = x
        self.xcell_id = xcell_id
        self.t_vae = t_vae
        self.e_vae = e_vae
        self.xtime = set_timeax
        self.xfreq = set_freqax
        self.transform = transforms.Compose([transforms.ToTensor()])
      
    def __len__(self):
        return(self.x.shape[0])
      
    def __getitem__(self,idx):
        idx_x = self.x[idx]
        qz_mu, qz_logvar = self.t_vae.enc_z(idx_x)
        qz_logvar = self.t_vae.softplus(qz_logvar)
        qz = dist.Normal(qz_mu, qz_logvar)
        t_z = qz.rsample()
        idx_xcell_id = np.load(self.xcell_id[idx])
        set_timeax = self.xtime
        set_freqax = self.xfreq
        idx_xcell_id = np.array(Image.fromarray(idx_xcell_id).resize((set_timeax,set_freqax)))
        idx_xcell_id = self.transform(idx_xcell_id)
        qz_mu, qz_logvar = self.e_vae.enc_z(idx_xcell_id)
        qz_logvar = self.e_vae.softplus(qz_logvar)
        qz = dist.Normal(qz_mu, qz_logvar)
        e_z = qz.rsample()
        e_z = e_z.squeeze()
        return(t_z,e_z)
