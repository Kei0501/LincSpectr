import utils
from dataset import VaeDataSet, ZDataSet
from modules import TscVAE, EscVAE, Linker
from funcs import t_elbo_loss, e_elbo_loss

file_names = collect_filename("./data_for_VAE/*")
cell_list = collect_cellname(file_names)
count_mat, cell_id = make_features(adata, './data_for_VAE/*.npy', './data_for_VAE/', "VAE/([^.]+)")
set_timeax = 128
set_freqax = 128
dataset = VaeDataSet(count_mat, cell_id)
val_x, val_xcell_id, val_xcell_names = split_dataset(dataset)
valid_list = make_validlist(val_xcell_names)

# tVAE training
x_dim = count_mat.size()[1]
t_vae = TscVAE(x_dim, xz_dim=10, enc_z_h_dim=50, dec_z_h_dim=50, num_enc_z_layers=2, num_dec_z_layers=2)
epoch_num = 350
loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_vae.to(device)
optimizer = torch.optim.Adam(t_vae.parameters(), lr=4.0e-3)

for epoch in tqdm(range(epoch_num)):
    tsc_vae.train()
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
loader = torch.utils.data.DataLoader(train_dataset, batch_siz=16, shuffle=True, pin_memory=True)
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

zdataset = ZDataSet(count_mat,cell_id)
train_zdataset,val_zdataset,test_zdataset = torch.utils.data.dataset.random_split(zdataset,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42))
val_tz, val_ez, val_zname = split_dataset(val_zdataset,test_ratio, val_ratio)
train_tz, train_ez, train_zname = split_dataset(train_zdataset,test_ratio, val_ratio)

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
