import torch
import torchvision
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np

def read_s2(path):
    bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
    patch_size = (1068,1068)
    
    current=[]
    for band in bands:
        with rio.open('{}/{}.tif'.format(path, band), 'r') as f:
            band_tensor = torch.from_numpy(f.read()/4e3).float()
    
            if band_tensor.shape[-2:] != patch_size:
                band_tensor = torchvision.transforms.functional.resize(band_tensor, patch_size)
    
            current.append(band_tensor)
    
    return torch.cat(current)

def plot_s2(s2, msi=False, rgb_channels=(3,2,1), cmap='cividis'):
    if not msi:
        plt.imshow(s2[...,rgb_channels,:,:].permute(1,2,0))
        plt.axis('off')
    else:
        n_channels = s2.shape[-3]
        rows = int(np.sqrt(n_channels))
        cols = int(np.ceil(n_channels/rows))

        for ch_idx in range(n_channels):
            plt.subplot(rows, cols, 1+ch_idx)
            plt.imshow(s2[...,ch_idx,:,:],cmap=cmap)
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)