import utils
from dataset import VaeDataSet, ZDataSet
from modules import TscVAE, EscVAE, Linker
from funcs import t_elbo_loss, e_elbo_loss

#input path & adata
file_names = collect_filename("./data_for_VAE/*")
cell_list = collect_cellname(file_names)
count_mat, cell_id = make_features(adata, './data_for_VAE/*.npy', './data_for_VAE/', "VAE/([^.]+)")

set_timeax = 128
set_freqax = 128
dataset = VaeDataSet(count_mat, cell_id)
val_x, val_xcell_id, val_xcell_names = split_dataset(dataset)

x_dim = count_mat.size()[1]
xz_dim, enc_z_h_dim, dec_z_h_dim, num_enc_z_layers, num_dec_z_layers = 10, 50, 50, 2, 2
# tVAE training
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
t_model = TscVAE(x_dim, xz_dim, enc_z_h_dim, dec_z_h_dim, num_enc_z_layers, num_dec_z_layers)
batch_size = 16
epoch_num = 350
lr = 4.0e-3
loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t_model.to(device)
optimizer = torch.optim.Adam(t_model.parameters(), lr=lr)

for epoch in tqdm(range(epoch_num)):
    t_model.train()
    for x, xcell_id, xcell_names in loader:
        x = x.to(device)
        optimizer.zero_grad()
        z, qz, ld =  t_model(x)
        loss = t_elbo_loss(qz, ld, x)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            val_x = val_x.to(device)
            t_model.eval()
            z, qz, ld =  t_model(val_x)
            loss = t_elbo_loss(qz, ld, val_x)
            print(f'loss at epoch {epoch} is {loss}')

i_dim = np.load(cell_id[0]).reshape(-1).shape[0]
xz_dim, enc_z_h_dim, dec_z_h_dim, num_enc_z_layers, num_dec_z_layers = 10, 500, 500, 2, 2

# eVAE training
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
e_model = EscVAE(i_dim, xz_dim, enc_z_h_dim, dec_z_h_dim, num_enc_z_layers, num_dec_z_layers)
batch_size = 16
epoch_num = 300
lr = 1.0e-4
loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
e_model.to(device)
optimizer = torch.optim.Adam(e_model.parameters(), lr=lr)

for epoch in tqdm(range(epoch_num)):
    e_model.train()
    for x, xcell_id, xcell_names in loader:
        xcell_id = xcell_id.to(device)
        optimizer.zero_grad()
        z, qz, ld_img =  e_model(xcell_id.view(-1, set_timeax*set_freqax))
        loss = e_elbo_loss(qz, ld_img, xcell_id.view(ld_img.size()))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            val_xcell_id = val_xcell_id.to(device)
            e_model.eval()
            z, qz, ld_img =  e_model(val_xcell_id.view(-1, set_timeax*set_freqax))
            loss = e_elbo_loss(qz, ld_img, val_xcell_id.view(ld_img.size()))
            print(f'loss at epoch {epoch} is {loss}')

zdataset = ZDataSet(count_mat,cell_id)
train_zdataset,val_zdataset,test_zdataset = torch.utils.data.dataset.random_split(zdataset,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42))
val_tz, val_ez, val_zname = split_dataset(val_zdataset,test_ratio, val_ratio)
train_tz, train_ez, train_zname = split_dataset(train_zdataset,test_ratio, val_ratio)

# connection model training
linkz_model = Linker(latent_dim=10)
batch_size = 512
epoch_num = 350
lr = 1.0e-4
z_loader = torch.utils.data.DataLoader(train_zdataset, batch_size=batch_size, shuffle=True, pin_memory=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
linkz_model.to(device)
optimizer = torch.optim.Adam(linkz_model.parameters(), lr=lr)

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


def make_umap(dataset):
    t_test, e_test = [], []
    for i in range(len(dataset)):
        t_test.append(dataset[i][0])
        e_test.append(dataset[i][1])
    test_x = torch.stack(t_test, dim = 0)
    test_xcell_id = torch.stack(e_test, dim = 0)
    test_x = test_x.to(device)
    test_xcell_id = test_xcell_id.to(device)
    e_model.to(device)
    with torch.no_grad():
        e_model.eval()
        z, qz, ld_img = e_model(test_xcell_id.view(-1,set_timeax*set_freqax))
    
    reducer = umap.UMAP(n_neighbors=15,min_dist=0.01)
    embedding = reducer.fit_transform(z.cpu().detach().numpy())
    
    sns.scatterplot(x = embedding[:,0],y = embedding[:,1],hue=adata.obs['RNA family'])
    plt.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
