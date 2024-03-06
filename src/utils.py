import glob
import numpy as np
from pynwb import NWBHDF5IO
from commons import get_time_voltage_current_currindex0
from ssqueezepy import ssq_cwt
import pandas as pd
import torch
import umap
from functorch import vmap
from functorch import vjp
import lpips
from sklearn.cluster import KMeans

def collect_filename(path):
  folders = glob.glob(path)
  file_names = []
  for folder in folders:
      files = glob.glob(folder + "/*")
      for file in files:
          file_names.append(file)
  return(file_names)

def collect_cellname(file_names):
  cell_list = []
  for cell in file_names:
      fixname = cell[45:53] + "_sample_" + cell[61:63]
      if fixname[17] == "_":
          fixname = fixname.rstrip("_")
      cell_list.append(fixname)
  return(cell_list)


def transform_efeatures(save_path, pick_voltage=30, set_timeax=128, set_freqax=128, set_parameter=1131):
  warnings.filterwarnings("ignore") # It complains about some namespaces, but it should work.
  io_ = NWBHDF5IO(file_names[set_parameter], 'r', load_namespaces=True)
  nwb = io_.read()
  time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(nwb)
  Twxo, Wxo, ssq_freqso, scaleso = ssq_cwt(voltage[:,pick_voltage])
  for i in range(len(file_names)):
      io_ = NWBHDF5IO(file_names[i], 'r', load_namespaces=True)
      nwb = io_.read()
      time, voltage, current, curr_index_0 = get_time_voltage_current_currindex0(nwb)
      if voltage.shape[1] >pick_voltage:
          Twxo, Wxo, *_ = ssq_cwt(voltage[:,pick_voltage],ssq_freqs = ssq_freqso,scales = scaleso)
          image_outcome = np.abs(Wxo)
          image_length = len(image_outcome[0])
          test_length = int(image_length*0.01)
          search_outlier = np.sum(image_outcome[:,:test_length],axis=1)
          search_outlier = search_outlier/len(search_outlier)
          calc_mean = np.mean(search_outlier)
          calc_std = np.std(search_outlier)
          limit_low=calc_mean-calc_std
          limit_high=calc_mean+calc_std
        
          except_freq = []
          for j in range(len(search_outlier)):
              if (search_outlier[j] > limit_high and j>230) or (search_outlier[j] < limit_low and j>230):
                  except_freq.append(j)
          pick_freq = np.zeros(len(search_outlier))
          np.put(pick_freq,except_freq,1)
          delete_freq = search_outlier * pick_freq
          delete_array = np.tile(delete_freq,(image_length,1)).T
          picked_outcome = image_outcome - delete_array
          processed_outcome = picked_outcome[75:,:20000]
          resized_outcome = np.array(Image.fromarray(processed_outcome).resize((set_timeax,set_freqax)))
          np.save(save_path + cell_list[i] + '.npy', resized_outcome)
  
      else:
          continue


def make_features(adata, data_path, file_path, extract_part):
  cells = pd.Series(glob.glob(data_path)).str.extract(extract_part).iloc[:,0].values
  delete_cells = []
  for cell in cells:
      if (cell in adata.obs_names) == True:
          delete_cells.append(cell)
  adata = adata[delete_cells]
  count_mat = torch.Tensor(adata[adata.obs_names,adata.var.highly_variable].layers['count'].toarray())

  cell_id = []
  for i in range(len(delete_cells)):
      cell_id.append(file_path + delete_cells[i] + '.npy')
  return(count_mat, cell_id)


def get_time_voltage_current_currindex0(nwb):
    df = nwb.sweep_table.to_dataframe()
    voltage = np.zeros((len(df['series'][0][0].data[:]), int((df.shape[0]+1)/2)))
    time = np.arange(len(df['series'][0][0].data[:]))/df['series'][0][0].rate
    voltage[:, 0] = df['series'][0][0].data[:]
    current_initial = df['series'][1][0].data[12000]*df['series'][1][0].conversion
    curr_index_0 = int(-current_initial/20) # index of zero current stimulation
    current = np.linspace(current_initial, (int((df.shape[0]+1)/2)-1)*20+current_initial, \
                         int((df.shape[0]+1)/2))
    for i in range(curr_index_0):   # Find all voltage traces from minimum to 0 current stimulation
        voltage[:, i+1] = df['series'][0::2][(i+1)*2][0].data[:]
    for i in range(curr_index_0, int((df.shape[0]+1)/2)-1):   # Find all voltage traces from 0 to highest current stimulation
        voltage[:, i+1] = df['series'][1::2][i*2+1][0].data[:]
    voltage[:, curr_index_0] = df.loc[curr_index_0*2][0][0].data[:]    # Find voltage trace for 0 current stimulation
    return time, voltage, current, curr_index_0


def plot_volatge(voltage):
    plt.plot(voltage[:,pick_voltage])    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)    
    plt.show


def embed_z(z, n_neighbors, min_dist):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    z_embed = reducer.fit_transform(z.cpu().detach().numpy())
    return(z_embed)


def split_dataset(dataset,test_ratio,val_ratio):
  total_size = len(dataset)
  test_size = int(total_size * test_ratio)
  val_size = int(total_size * val_ratio)
  train_size = total_size - test_size - val_size
  train_dataset,val_dataset,test_dataset = torch.utils.data.dataset.random_split(dataset,[train_size,val_size,test_size],generator=torch.Generator().manual_seed(42))
  t_val, e_val, val_xcell_names = [], [], []
  for i in range(len(val_dataset)):
      t_val.append(val_dataset[i][0])
      e_val.append(val_dataset[i][1])
      val_xcell_names.append(val_dataset[i][2])
  val_x = torch.stack(t_val, dim = 0)
  val_xcell_id = torch.stack(e_val, dim = 0)
  return(val_x, val_xcell_id, val_xcell_names)

def celltype_list(cell_list):
  cell_data = adata[cell_list]
  Vip, Lamp5, Pvalb, Sst, ET, IT, CT = [], [], [], [], [], [], []
  
  for i in range(len(valid_data.obs)):
      if cell_data.obs["RNA family"][i] == "Vip":
          Vip.append(cell_data.obs_names[i])
      if cell_data.obs["RNA family"][i] == "Lamp5":
          Lamp5.append(cell_data.obs_names[i])
      if cell_data.obs["RNA family"][i] == "Pvalb":
          Pvalb.append(cell_data.obs_names[i])
      if cell_data.obs["RNA family"][i] == "Sst":
          Sst.append(cell_data.obs_names[i])
      if cell_data.obs["RNA family"][i] == "ET":
          ET.append(cell_data.obs_names[i])
      if cell_data.obs["RNA family"][i] == "IT":
          IT.append(cell_data.obs_names[i])
      if cell_data.obs["RNA family"][i] == "CT":
          CT.append(cell_data.obs_names[i])
      else:
          continue
  return(Vip, Lamp5, Pvalb, Sst, ET, IT, CT)


def GetMedoid(vX):
  vMean = np.mean(vX, axis=0)
  min_number = np.argmin([sum((x - vMean)**2) for x in vX])
  return vX[min_number], min_number  


def average_expression(adata, cell_type):
  gene_sum = torch.zeros(count_mat.size()[1])
  counts = 0
  for i in range(len(adata.obs)):
      if adata.obs["RNA family"][i] == cell_type:
          obsname = adata.obs_names[i]
          bdata = adata[obsname]
          pick_count = torch.Tensor(adata[bdata.obs_names,bdata.var.highly_variable].layers['count'].toarray())
          pick_count = pick_count.squeeze()
          gene_sum = gene_sum + pick_count
          counts = counts + 1
  avr_express = torch.div(gene_sum,counts)
  return(avr_express)


def calc_lpips(adata, validlist, path, sample_path):  
  loss_fn_alex = lpips.LPIPS(net='alex')
  VAE_list, baseline_list = [], []
  for valid_sample in validlist:
      read_image = np.load(path + valid_sample +  ".npy")
      right_image = read_image.reshape(np.load(sample_path).shape)
      right_image = torch.from_numpy(right_image.astype(np.float32)).clone()
      right_sum = torch.sum(right_image)
      testdata = adata[valid_sample]
      testcount_mat = torch.Tensor(testdata[testdata.obs_names,testdata.var.highly_variable].layers['count'].toarray())
      testcount_mat = torch.ravel(testcount_mat)
      test_image = LincSpectr(testcount_mat)
      predicted_image = test_image.reshape(np.load(sample_path).shape)
      predict_sum = torch.sum(predicted_image)

      LPIPS_vae = loss_fn_alex(right_image,predicted_image)
      VAE_list.append(LPIPS_vae)
      LPIPS_baseline = loss_fn_alex(right_image,rightimage_avr)
      baseline_list.append(LPIPS_baseline)
  return(VAE_list, baseline_list)


def inverse_analysis(avr_express, N, image_shape):
  image_size = image_shape[0] * image_shape[1]
  I_use = torch.eye(image_size,image_size)
  _, fn_vjp = vjp(lambda vx:LincSpectr(vx), avr_express)
  jacobian_mat, = vmap(fn_vjp)(I_use)
  u, s, vh = torch.linalg.svd(jacobian_mat)
  u_pick = u[:,0]
  vT = torch.conj(vh)
  v_pick = vT[0]
  v_pick = v_pick.to('cpu').detach().numpy().copy()
  top_genes = []
  for i in range(N):
      j = i + 1
      gene_pos_number = np.where(v_pick==np.sort(v_pick)[-j])[0][0]
      top_genename = adata[adata.obs_names,adata.var.highly_variable].var_names[gene_pos_number]
      top_genes.append(top_genename)
  return(u_pick, v_pivk, top_genes)


def kmeans_cluster(embedding):
  cluster_data = pd.DataFrame(embedding)
  km = KMeans(n_clusters=7, max_iter=30)
  kmh = km.fit_predict(embedding)
  cluster_data['kmh'] = kmh
  for i in np.sort(cluster_data['kmh'].unique()):
      plt.scatter(cluster_data[cluster_data['kmh']==i][0], cluster_data[cluster_data['kmh']==i][1], label=f'cluster{i}')
  plt.legend()


def make_umap(z, n_neighbors=15, min_dist=0.01):
    reducer = umap.UMAP(n_neighbors,min_dist)
    embedding = reducer.fit_transform(z.cpu().detach().numpy())
    sns.scatterplot(x = embedding[:,0],y = embedding[:,1],hue=adata.obs['RNA family'])
    plt.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
