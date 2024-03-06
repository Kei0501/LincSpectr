import glob
import numpy as np
from pynwb import NWBHDF5IO
from commons import get_time_voltage_current_currindex0
from ssqueezepy import ssq_cwt
import pandas as pd
import torch
import umap

def collact_filename(path):
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
