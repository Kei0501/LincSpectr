import scanpy as sc
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import utils
from dataset import VaeDataSet, ZDataSet
from modules import TscVAE, EscVAE, Linker
from funcs import t_elbo_loss, e_elbo_loss

#prepare transcriptome data
#add RNA_family information and remove low quality data
adata = sc.read_csv('./m1_patchseq_exon_counts.csv')
plus_data = pd.read_table('./m1_patchseq_meta_data.csv',sep='\t',index_col=1)
adata = adata.T
adata.obs = plus_data.loc[adata.obs_names]
adata = adata[adata.obs["RNA family"] != "low quality"]
adata.layers['count'] = adata.X
sc.pp.filter_cells(adata,min_counts=100)
sc.pp.filter_genes(adata,min_cells=10)
sc.pp.normalize_total(adata,target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=2000)

count_mat, cell_id, adata = utils.make_features(adata, './data_for_VAE/*.npy', './data_for_VAE/', "VAE/([^.]+)")
set_timeax = 128
set_freqax = 128
dataset = VaeDataSet(count_mat, cell_id)
val_x, val_xcell_id, val_xcell_names, train_dataset,val_dataset,test_dataset = utils.split_dataset(dataset,test_ratio=0.05,val_ratio=0.1)
valid_list = utils.make_validlist(val_xcell_names)

# tVAE training
x_dim = count_mat.size()[1]
t_vae = TscVAE(x_dim, xz_dim=10, enc_z_h_dim=50, dec_z_h_dim=50, num_enc_z_layers=2, num_dec_z_layers=2)
epoch_num = 350
loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_vae.to(device)
optimizer = torch.optim.Adam(t_vae.parameters(), lr=4.0e-3)

for epoch in tqdm(range(epoch_num)):
    t_vae.train()
    for x, xcell_id, xcell_names in loader:
        x = x.to(device)
        optimizer.zero_grad()
        z, qz, ld =  t_vae(x)
        loss = t_elbo_loss(qz, ld, x)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            val_x = val_x.to(device)
            t_vae.eval()
            z, qz, ld =  t_vae(val_x)
            loss = t_elbo_loss(qz, ld, val_x)
            print(f'loss at epoch {epoch} is {loss}')

# eVAE training
i_dim = np.load(cell_id[0]).reshape(-1).shape[0]
e_vae = EscVAE(i_dim, xz_dim=10, enc_z_h_dim=500, dec_z_h_dim=500, num_enc_z_layers=2, num_dec_z_layers=2)
epoch_num = 300
loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
e_vae.to(device)
optimizer = torch.optim.Adam(e_vae.parameters(), lr=1.0e-4)

for epoch in tqdm(range(epoch_num)):
    e_vae.train()
    for x, xcell_id, xcell_names in loader:
        xcell_id = xcell_id.to(device)
        optimizer.zero_grad()
        z, qz, ld_img =  e_vae(xcell_id.view(-1, set_timeax*set_freqax))
        loss = e_elbo_loss(qz, ld_img, xcell_id.view(ld_img.size()))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            val_xcell_id = val_xcell_id.to(device)
            e_vae.eval()
            z, qz, ld_img =  e_vae(val_xcell_id.view(-1, set_timeax*set_freqax))
            loss = e_elbo_loss(qz, ld_img, val_xcell_id.view(ld_img.size()))
            print(f'loss at epoch {epoch} is {loss}')

zdataset = ZDataSet(count_mat,cell_id,t_vae,e_vae)
test_ratio = 0.05
val_ratio = 0.1

total_size = len(zdataset)
test_size = int(total_size * test_ratio)
val_size = int(total_size * val_ratio)
train_size = total_size - test_size - val_size
train_zdataset,val_zdataset,test_zdataset = torch.utils.data.dataset.random_split(zdataset,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42))

tz_features = []
ez_features = []
for i in range(len(val_zdataset)):
    plus_tz = val_zdataset[i][0]
    tz_features.append(plus_tz)
    plus_ez = val_zdataset[i][1]
    ez_features.append(plus_ez)
val_tz = torch.stack(tz_features)
val_ez = torch.stack(ez_features)

ez_train_features = []
for i in tqdm(range(len(train_zdataset))):
    train_ez = train_zdataset[i][1]
    ez_train_features.append(train_ez)
val_ez_train = torch.stack(ez_train_features)

# connection model training
linkz_model = Linker(latent_dim=10)
epoch_num = 350
z_loader = torch.utils.data.DataLoader(train_zdataset, batch_size=512, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
linkz_model.to(device)
optimizer = torch.optim.Adam(linkz_model.parameters(), lr=1.0e-4)

for epoch in tqdm(range(epoch_num)):
    linkz_model.train()
    for t_z, e_z  in z_loader:
        xt_z = t_z.to(device)
        xe_z = e_z.to(device)
        optimizer.zero_grad()
        lp =  linkz_model(xt_z,xe_z)
        lp = torch.diag(lp,0)
        loss = lp.sum()
        loss.backward(retain_graph=True)
        optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            val_xt_z = val_tz.to(device)
            val_xe_z = val_ez.to(device)
            linkz_model.eval()
            lp = linkz_model(val_xt_z,val_xe_z)
            lp = torch.diag(lp,0)
            loss = lp.sum()
            print(f'loss at epoch {epoch} is {loss}')
