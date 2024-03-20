import glob
import numpy as np
from pynwb import NWBHDF5IO
from ssqueezepy import ssq_cwt
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
import pandas as pd
import torch
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from functorch import vmap
from functorch import vjp
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


def transform_efeatures(file_names, save_path, cell_list, pick_voltage=30, set_timeax=128, set_freqax=128, set_parameter=1131, cut_freq=230, start_frame=75, end_frame=20000):
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
              if (search_outlier[j] > limit_high and j>cut_freq) or (search_outlier[j] < limit_low and j>cut_freq):
                  except_freq.append(j)
          pick_freq = np.zeros(len(search_outlier))
          np.put(pick_freq,except_freq,1)
          delete_freq = search_outlier * pick_freq
          delete_array = np.tile(delete_freq,(image_length,1)).T
          picked_outcome = image_outcome - delete_array
          processed_outcome = picked_outcome[start_frame:,:end_frame]
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
  return(count_mat, cell_id, adata)


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
  return(val_x, val_xcell_id, val_xcell_names,train_dataset,val_dataset,test_dataset)


def make_validlist(val_xcell_names):
  valid_list = []
  for val_cell in val_xcell_names:
      valname = val_cell[15:23] + "_sample_" + val_cell[31:33]
      if valname[17] == ".":
          valname = valname.rstrip(".")  
      valid_list.append(valname)
  return(valid_list)


def celltype_list(adata,cell_list):
  cell_data = adata[cell_list]
  Vip, Lamp5, Pvalb, Sst, ET, IT, CT = [], [], [], [], [], [], []
  
  for i in range(len(cell_data.obs)):
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


def average_expression(adata, count_mat, cell_type):
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


def inverse_analysis(avr_express,adata,LincSpectr,t_vae,e_vae,linkz_model,train_ez,val_ez,N, image_shape, pick_num=0):
  image_size = image_shape[0] * image_shape[1]
  I_use = torch.eye(image_size,image_size)
  _, fn_vjp = vjp(lambda vx:LincSpectr(vx,t_vae,e_vae,linkz_model,train_ez,val_ez), avr_express)
  jacobian_mat, = vmap(fn_vjp)(I_use)
  u, s, vh = torch.linalg.svd(jacobian_mat)
  u_pick = u[:,pick_num]
  vT = torch.conj(vh)
  v_pick = vT[pick_num]
  v_pick = v_pick.to('cpu').detach().numpy().copy()
  top_genes, top_expression = [], []
  for i in range(N):
      j = i + 1
      gene_pos_number = np.where(v_pick==np.sort(v_pick)[-j])[0][0]
      top_genename = adata[adata.obs_names,adata.var.highly_variable].var_names[gene_pos_number]
      gene_count = v_pick[gene_pos_number]
      top_genes.append(top_genename)
      top_expression.append(gene_count)
  return(u_pick, top_genes, top_expression)


def load_realimg(cell_name,set_timeax=128,set_freqax=128):
    cell_idx = './data_for_VAE/' + cell_name + '.npy'
    real_image = np.load(cell_idx)
    real_image = np.array(Image.fromarray(real_image).resize((set_timeax,set_freqax)))
    return(real_image)


def predict_img(adata,cell_name,sample_file,LincSpectr,t_vae,e_vae,linkz_model,train_ez,val_ez):
    testdata = adata[cell_name]
    testcount_mat = torch.Tensor(testdata[testdata.obs_names,testdata.var.highly_variable].layers['count'].toarray())
    testcount_mat = testcount_mat.squeeze()
    test_image = LincSpectr(testcount_mat,t_vae,e_vae,linkz_model,train_ez,val_ez)
    reshape_image = test_image.reshape(np.load(sample_file).shape)
    predicted_image = reshape_image.to('cpu').detach().numpy().copy()
    return(predicted_image)


def show_prediction(sample_file,cell_name1, cell_name2,adata,LincSpectr,t_vae,e_vae,linkz_model,train_ez,val_ez):
    fig = plt.figure()
    X = 2
    Y = 2

    imgplot = 1
    ax1 = fig.add_subplot(X, Y, imgplot)
    check_image = load_realimg(cell_name1)
    ax1.set_title("real_1",fontsize=10)
    plt.imshow(check_image, aspect='auto', cmap='turbo', vmin=0)

    imgplot = 2
    ax1 = fig.add_subplot(X, Y, imgplot)
    predicted_image = predict_img(adata,cell_name1,sample_file,LincSpectr,t_vae,e_vae,linkz_model,train_ez,val_ez)
    ax1.set_title("predict_1",fontsize=10)
    plt.imshow(predicted_image, aspect='auto', cmap='turbo', vmin=0)

    imgplot = 3
    ax1 = fig.add_subplot(X, Y, imgplot)
    check_image2 = load_realimg(cell_name2)
    ax1.set_title("real_2",fontsize=10)
    plt.imshow(check_image2, aspect='auto', cmap='turbo', vmin=0)

    img2plot =  4
    ax2 = fig.add_subplot(X, Y, img2plot)
    predicted_image2 = predict_img(adata,cell_name2,sample_file,LincSpectr,t_vae,e_vae,linkz_model,train_ez,val_ez)
    ax2.set_title("predict_2",fontsize=10)
    plt.imshow(predicted_image2, aspect='auto', cmap='turbo', vmin=0)


def calc_lpips(adata, validlist, path, sample_path):  
  loss_fn_alex = lpips.LPIPS(net='alex')
  vae_list, baseline_list = [], []
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

      lpips_vae = loss_fn_alex(right_image,predicted_image)
      vae_list.append(lpips_vae)
      lpips_baseline = loss_fn_alex(right_image,rightimage_avr)
      baseline_list.append(lpips_baseline)
  return(vae_list, baseline_list)


def kmeans_cluster(embedding):
  cluster_data = pd.DataFrame(embedding)
  km = KMeans(n_clusters=7, max_iter=30)
  kmh = km.fit_predict(embedding)
  cluster_data['kmh'] = kmh
  for i in np.sort(cluster_data['kmh'].unique()):
      plt.scatter(cluster_data[cluster_data['kmh']==i][0], cluster_data[cluster_data['kmh']==i][1], label=f'cluster{i}')
  plt.legend()
