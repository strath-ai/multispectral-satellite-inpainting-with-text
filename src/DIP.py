import pytorch_lightning as pl
from .backbone.SkipNetwork import *

import torch
import torch.nn.functional as F

class ShellDataset:
    def __init__(self, iterations = 100):
        self.iterations = iterations
    def __len__(self):
        return self.iterations
    def __getitem__(self,idx):
        return 0

class DIP(pl.LightningModule):

    def __init__(self,
                 target_channels = 3,
                 latent_seed = 'meshgrid',
                 sigmoid_output = False,
                 lr = 2e-2,
                 regularization_noise_std = 0.0,
                 epoch_steps = 1000,
                 loss_trace=False,
                 logging = False
                ):
        super().__init__()
        
        self.input = None        
        self.target = None
        self.mask = None
        self.latent_seed = latent_seed
        self.sigmoid_output = sigmoid_output
        self.logging = logging
        self.epoch_steps = epoch_steps
        self.target_channels = target_channels
        self.regularization_noise_std = regularization_noise_std
        self.lr = lr
        
        if loss_trace:
            self.loss_trace=[]
        else:
            self.loss_trace=None

        if isinstance(self.latent_seed, list) or isinstance(self.latent_seed, tuple):
            self.input_depth = self.latent_seed[1]
        elif self.latent_seed == 'meshgrid':
            self.input_depth = 2
        else:
            self.input_depth = np.sum(self.target_channels)
        
        # Model
        self.init_model()   
            
        self.train_dataset = ShellDataset(self.epoch_steps)        
       
    def image_check(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        if len(x.shape) < 4:
            x = torch.unsqueeze(x, 0)  
        if np.argmin(x.shape[1:]) == 2:
            x = x.permute(0, 3, 1, 2)
        return x.to(self.device)
    
    def init_model(self):
        # source from original DIP github repository
        self.model = SkipNetwork(in_channels=self.input_depth,
                                 out_channels=int(np.sum(self.target_channels)),
                                 hidden_dims_down = [16, 32, 64, 128, 128, 128],
                                 hidden_dims_up = [16, 32, 64, 128, 128, 128],
                                 hidden_dims_skip = [0, 0, 0, 0, 0, 0],
                                 filter_size_down = 3,
                                 filter_size_up = 5,
                                 filter_skip_size = 0,
                                 sigmoid_output = self.sigmoid_output,
                                 bias = True,
                                 padding_mode='zero',
                                 upsample_mode='nearest',
                                 downsample_mode='stride',
                                 act_fun='LeakyReLU',
                                 need1x1_up=True
                                )    
    
    def set_input(self, x):
        if type(x) is not list:
            self.input = self.image_check(x)
        else:
            x_c = []
            for x_i in x:
                x_i_c = self.image_check(x_i)
                x_c.append(x_i_c)
            self.input = torch.cat(x_c, 1).float()
            
    def set_target(self, *targets):
        
        # set target
        self.target=[]
        self.target_channels=[]
        for target in targets:
            self.target_channels.append(target.shape[-3])
            self.target.append(self.image_check(target))
        self.target = torch.cat(self.target, -3)
       
        # at the same time, initialize input
        if isinstance(self.latent_seed, list) or isinstance(self.latent_seed, tuple):
            self.set_input(self.latent_seed[0]*torch.rand(self.latent_seed[1], *self.target.shape[-2:]))
            
        elif self.latent_seed == 'meshgrid':
            X,Y = torch.meshgrid(torch.arange(0, 1, 1/self.target.shape[-2]),
                                 torch.arange(0, 1, 1/self.target.shape[-1])
                                )
            meshgrid = torch.cat([X[None,:], Y[None,:]])            
            self.set_input(meshgrid)
            
        else:
            self.set_input(self.target) 
            
        # reinit model
        self.init_model()   
        
    def set_mask(self, *masks): 

        assert len(masks)==len(self.target_channels)
        
        self.mask=[]
        for idx,mask in enumerate(masks):
            if mask is None:
                self.mask.append(torch.ones(1,self.target_channels[idx],*self.target.shape[-2:]))
            else:
                if len(mask.shape)==2:
                    flat_mask = torch.tensor(mask).view(1,*mask.shape[-2:])
                    self.mask.append(torch.stack(self.target_channels[idx]*[flat_mask.clone()], 1))
                else:
                    self.mask.append(mask)

        self.mask = torch.cat(self.mask, -3).bool()
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset)
        return train_loader
    
    def output(self):
        out = self.forward(self.input.to(self.device)).detach()

        if type(self.target_channels) is list:
            out_list = []
            offset_ch = 0
            for ch in self.target_channels:
                out_list.append(out[0,offset_ch:offset_ch+ch,...].cpu())
                offset_ch += ch
            return out_list
        else:
            return out[0].cpu()

    def forward(self, input):
        return self.model.forward(input)            
        
    def get_loss(self, output, target, mask = None):
        
        loss = F.mse_loss(output, target, reduction = 'none')
        
        if mask is not None:
            if len(mask.shape) == len(loss.shape):
                loss = loss[mask == 1.0]
            else:
                loss = loss[:, mask==1.0]
        
        return loss.mean()
    
    def on_fit_start(self):
        self.input = self.input.to(self.device)
        self.target = self.target.to(self.device)

    def training_step(self, batch, batch_idx):
            
        if self.regularization_noise_std != 0.0:
            input = self.input + self.regularization_noise_std*torch.randn(self.input.shape, device=self.input.device)
        else:
            input = self.input
            
        out = self.forward(input)
        loss = self.get_loss(out, self.target, self.mask)
        
        if self.loss_trace is not None:
            self.loss_trace.append(float(loss['loss'].item()))
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)