import torch
import numpy as np
import torch.distributions as dist

class VaeDataSet(torch.utils.data.Dataset):
    def __init__(self,x,xcell_id,transform=None,pre_transform=None):
        self.x = x
        self.xcell_id = xcell_id
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return(self.x.shape[0])
      
    def __getitem__(self,idx):
        idx_x = self.x[idx]
        idx_xcell_id = np.load(self.xcell_id[idx])
        idx_xcell_id = np.array(Image.fromarray(idx_xcell_id).resize((set_timeax,set_freqax)))
        idx_xcell_id = self.transform(idx_xcell_id)
        idx_xcell_name = self.xcell_id[idx]
        return(idx_x,idx_xcell_id,idx_xcell_name)

    def normalize(self,img):
        return (img-img.min())/(img.max()-img.min())


class ZDataSet(torch.utils.data.Dataset):
    def __init__(self,x,xcell_id,transform=None,pre_transform=None):
        self.x = x
        self.xcell_id = xcell_id
        self.transform = transforms.Compose([transforms.ToTensor()])
      
    def __len__(self):
        return(self.x.shape[0])
      
    def __getitem__(self,idx):
        idx_x = self.x[idx]
        qz_mu, qz_logvar = t_vae.enc_z(idx_x)
        qz_logvar = t_vae.softplus(qz_logvar)
        qz = dist.Normal(qz_mu, qz_logvar)
        t_z = qz.rsample()
        idx_xcell_id = np.load(self.xcell_id[idx])
        idx_xcell_id = np.array(Image.fromarray(idx_xcell_id).resize((set_timeax,set_freqax)))
        idx_xcell_id = self.transform(idx_xcell_id)
        qz_mu, qz_logvar = e_model.enc_z(idx_xcell_id)
        qz_logvar = e_model.softplus(qz_logvar)
        qz = dist.Normal(qz_mu, qz_logvar)
        e_z = qz.rsample()
        e_z = e_z.squeeze()
        return(t_z,e_z)

    def normalize(self,img):
        return (img-img.min())/(img.max()-img.min())
