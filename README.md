# LincSpectr(Deep-Learning model integrating single cell Specific time-frequency electrophysiological characteristics and transcriptomes)

**LincSpectr** is a deep generative model interpretably integrating single-cell time-frequency characteristics transformed from electrophysiological data with transcriptomic features.


Teppei Shimamura's lab, Tokyo Medical and Dental university at Tokyo and Nagoya University at Nagoya

Yasuhiro Kojima's lab, National Cancer Center Research Institute at Tokyo

Developed by Kazuki Furumichi

# Installation
You can use the latest development version from GitHub.

```
!git clone https://github.com/Kei0501/LincSpectr
```

# Usage
You need to prepare [AnnData objects](https://anndata.readthedocs.io/en/latest/) which includes raw count matrix of gene expression for single cell and CWT transformed electrophysiological data respectively. You can see the usage in [IPython Notebook](https://github.com/Kei0501/LincSpectr/blob/main/tutorial/LincSpectr_tutorial.ipynb).

