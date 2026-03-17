#!/usr/bin/env python
# coding: utf-8

# # Train MLP from raw data

# # TODO:
# 
# 1. Benchmark ODE or Algebric model
# 2. Try more complex MLP. How many layers? How many nodes per layer? Best so far: 1.900
# 3. Log other metrics such as mean absolute error (MAE), the root mean square error (RMSE), coefficient of determination (R2), and relative error (delta) ??

# In[1]:


import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set deterministic behaviour
pl.seed_everything(1234)


# In[2]:


# PARAMETERS

batch_size = 64
learning_rate = 1e-4
num_workers = 7


# In[3]:


# Load data
Retau_mean = 1.02003270E+003
utau_mean  = 4.97576926E-002
deltav = 0.5 / Retau_mean
mu = deltav * utau_mean
rho = 1.
tauwall_mean = rho * (utau_mean ** 2)

    #data = np.load('../../../../DATA/dataset_retau_1000_tauwall_target.npy')
    #data = np.load('new_dataset.npy')
    # In[4]:
train = np.load('dataset_train_utau30flut_nondim_dns.npy')
if __name__ == "__main__":

    test = np.load('dataset_test_utau30flut_nondim_dns.npy')
    valid = np.load('dataset_valid_utau30flut_nondim_dns.npy')
    
y_train = train[:,0]
if __name__ == "__main__":

    X_train = train[:,1:]

    y_test = test[:,0]
    X_test = test[:,1:]

    y_val = valid[:,0]
    X_val = valid[:,1:]
    
    import seaborn as sns
    
    from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    #normalizzo il target
    #epsilon = 0.1 * data[:,0].min()
    #y_train_log = np.log(y_train + epsilon)
    #y_test_log = np.log(y_test + epsilon)
    #y_val_log = np.log(y_val + epsilon)
    
    #rb = RobustScaler().fit(y_train.reshape(-1,1))
    #y_train_log = rb.fit_transform(y_train.reshape(-1,1)).ravel()
    #y_test_log = rb.fit_transform(y_test.reshape(-1,1)).ravel()
    #y_val_log = rb.fit_transform(y_val.reshape(-1,1)).ravel()
   # pt_target = PowerTransformer(method='yeo-johnson').fit(y_train.reshape(-1,1))
   # y_train_log = pt_target.transform(y_train.reshape(-1,1)).ravel()
   # y_test_log = pt_target.transform(y_test.reshape(-1,1)).ravel()
   # y_val_log = pt_target.transform(y_val.reshape(-1,1)).ravel()
    #setto gli scaler per la normalizzazione delle feature
    ss_target = PowerTransformer('yeo-johnson').fit(y_train.reshape(-1,1))
    #ss_target = StandardScaler().fit(y_train.reshape(-1,1))
    y_train = ss_target.transform(y_train.reshape(-1,1)).ravel()
    y_test = ss_target.transform(y_test.reshape(-1,1)).ravel()
    y_val = ss_target.transform(y_val.reshape(-1,1)).ravel()
    ss_w60 = PowerTransformer().fit(X_train[:,2].reshape(-1,1))
    ss_w120 = PowerTransformer().fit(X_train[:,5].reshape(-1,1))
  #  ss_w60 = StandardScaler().fit(X_train[:,2].reshape(-1,1))
  #  ss_w120 = StandardScaler().fit(X_train[:,5].reshape(-1,1))
  #  mm = MinMaxScaler(feature_range = (-1,1))
  #  mm_u60 = mm.fit(X_train[:,0].reshape(-1,1))
  #  mm_v60 = mm.fit(X_train[:,1].reshape(-1,1))
  #  mm_w60 = mm.fit(X_train[:,2].reshape(-1,1))
  #  mm_u120 = mm.fit(X_train[:,3].reshape(-1,1))
  #  mm_v120 = mm.fit(X_train[:,4].reshape(-1,1))
  #  mm_w120 = mm.fit(X_train[:,5].reshape(-1,1))

  #  ss_u60 = StandardScaler().fit(X_train[:,0].reshape(-1,1))
  #  ss_v60 = StandardScaler().fit(X_train[:,1].reshape(-1,1))
  #  ss_u120 = StandardScaler().fit(X_train[:,3].reshape(-1,1))
  #  ss_v120 = StandardScaler().fit(X_train[:,4].reshape(-1,1))
    
    ss_u60  = PowerTransformer('yeo-johnson').fit(X_train[:,0].reshape(-1,1))
    ss_v60  = PowerTransformer('yeo-johnson').fit(X_train[:,1].reshape(-1,1))
    ss_u120 = PowerTransformer('yeo-johnson').fit(X_train[:,3].reshape(-1,1))
    ss_v120 = PowerTransformer('yeo-johnson').fit(X_train[:,4].reshape(-1,1))

    import joblib
   # joblib.dump(pt_target, 'pt_target_splitted.pkl')
    joblib.dump(ss_u60, 'ss_u120_utau120_dns.pkl')
    joblib.dump(ss_v60, 'ss_v120_utau120_dns.pkl')
    joblib.dump(ss_w60, 'ss_w120_utau120_dns.pkl')
    joblib.dump(ss_u120, 'ss_u30_utau30_dns.pkl')
    joblib.dump(ss_v120, 'ss_v30_utau30_dns.pkl')
    joblib.dump(ss_w120, 'ss_w30_utau30_dns.pkl')
    joblib.dump(ss_target, 'ss_target_utau60flut_dns_nondim_dns.pkl')
  #  joblib.dump(mm_u60, 'mm_u60_splitted.pkl')
  #  joblib.dump(mm_v60, 'mm_v60_splitted.pkl')
  #  joblib.dump(mm_w60, 'mm_w60_splitted.pkl')
  #  joblib.dump(mm_u120, 'mm_u120_splitted.pkl')
  #  joblib.dump(mm_v120, 'mm_v120_splitted.pkl')
  #  joblib.dump(mm_w120, 'mm_w120_splitted.pkl')

    
    X_train_standardized = np.zeros((X_train.shape[0], X_train.shape[1]))
    X_test_standardized = np.zeros((X_test.shape[0], X_test.shape[1]))
    X_val_standardized = np.zeros((X_val.shape[0], X_val.shape[1]))
    #normalizzo le features e le porto tutte in un range -1 e 1 buono per ReLU
    X_train_standardized[:,0] = ss_u60.transform(X_train[:,0].reshape(-1,1)).ravel()
    X_test_standardized[:,0] = ss_u60.transform(X_test[:,0].reshape(-1,1)).ravel()
    X_val_standardized[:,0] = ss_u60.transform(X_val[:,0].reshape(-1,1)).ravel()
  #  X_train_standardized[:,0] = mm_u60.transform(X_train_standardized[:,0].reshape(-1,1)).ravel()
  #  X_test_standardized[:,0] = mm_u60.transform(X_test_standardized[:,0].reshape(-1,1)).ravel()
  #  X_val_standardized[:,0] = mm_u60.transform(X_val_standardized[:,0].reshape(-1,1)).ravel()
    
    X_train_standardized[:,1] = ss_v60.transform(X_train[:,1].reshape(-1,1)).ravel()
    X_test_standardized[:,1] = ss_v60.transform(X_test[:,1].reshape(-1,1)).ravel()
    X_val_standardized[:,1] = ss_v60.transform(X_val[:,1].reshape(-1,1)).ravel()
  #  X_train_standardized[:,1] = mm_v60.transform(X_train_standardized[:,1].reshape(-1,1)).ravel()
  #  X_test_standardized[:,1] = mm_v60.transform(X_test_standardized[:,1].reshape(-1,1)).ravel()
  #  X_val_standardized[:,1] = mm_v60.transform(X_val_standardized[:,1].reshape(-1,1)).ravel()
    
    X_train_standardized[:,3] = ss_u120.transform(X_train[:,3].reshape(-1,1)).ravel()
    X_test_standardized[:,3] = ss_u120.transform(X_test[:,3].reshape(-1,1)).ravel()
    X_val_standardized[:,3] = ss_u120.transform(X_val[:,3].reshape(-1,1)).ravel()
  #  X_train_standardized[:,3] = mm_u120.transform(X_train_standardized[:,3].reshape(-1,1)).ravel()
  ##  X_test_standardized[:,3] = mm_u120.transform(X_test_standardized[:,3].reshape(-1,1)).ravel()
  ##  X_val_standardized[:,3] = mm_u120.transform(X_val_standardized[:,3].reshape(-1,1)).ravel()
  # 
    X_train_standardized[:,4] = ss_v120.transform(X_train[:,4].reshape(-1,1)).ravel()
    X_test_standardized[:,4] = ss_v120.transform(X_test[:,4].reshape(-1,1)).ravel()
    X_val_standardized[:,4] = ss_v120.transform(X_val[:,4].reshape(-1,1)).ravel()
  ##  X_train_standardized[:,4] = mm_v120.transform(X_train_standardized[:,4].reshape(-1,1)).ravel()
  ##  X_test_standardized[:,4] = mm_v120.transform(X_test_standardized[:,4].reshape(-1,1)).ravel()
  ##  X_val_standardized[:,4] = mm_v120.transform(X_val_standardized[:,4].reshape(-1,1)).ravel()
    
    X_train_standardized[:,2] = ss_w60.transform(X_train[:,2].reshape(-1,1)).ravel()
    X_test_standardized[:,2] = ss_w60.transform(X_test[:,2].reshape(-1,1)).ravel()
    X_val_standardized[:,2] = ss_w60.transform(X_val[:,2].reshape(-1,1)).ravel()
  #  X_train_standardized[:,2] = mm_w60.transform(X_train_standardized[:,2].reshape(-1,1)).ravel()
  #  X_test_standardized[:,2] = mm_w60.transform(X_test_standardized[:,2].reshape(-1,1)).ravel()
  #  X_val_standardized[:,2] = mm_w60.transform(X_val_standardized[:,2].reshape(-1,1)).ravel()
    
    X_train_standardized[:,5] = ss_w120.transform(X_train[:,5].reshape(-1,1)).ravel()
    X_test_standardized[:,5] = ss_w120.transform(X_test[:,5].reshape(-1,1)).ravel()
    X_val_standardized[:,5] = ss_w120.transform(X_val[:,5].reshape(-1,1)).ravel()
  #  X_train_standardized[:,5] = mm_w120.transform(X_train_standardized[:,5].reshape(-1,1)).ravel()
  #  X_test_standardized[:,5] = mm_w120.transform(X_test_standardized[:,5].reshape(-1,1)).ravel()
  #  X_val_standardized[:,5] = mm_w120.transform(X_val_standardized[:,5].reshape(-1,1)).ravel()
    
   # X_train_standardized[:,6] = np.log1p(X_train[:,6])
   # X_test_standardized[:,6] = np.log1p(X_test[:,6])
   # X_val_standardized[:,6] = np.log1p(X_val[:,6])
   # ss_utau_120_60 = StandardScaler().fit(X_train_standardized[:,6].reshape(-1,1))
   # ss_utau_120_60 = StandardScaler().fit(X_train[:,6].reshape(-1,1))
    ss_utau_120_60 = PowerTransformer('yeo-johnson').fit(X_train[:,6].reshape(-1,1))

    X_train_standardized[:,6] = ss_utau_120_60.transform(X_train[:,6].reshape(-1,1)).ravel()
    X_test_standardized[:,6] = ss_utau_120_60.transform(X_test[:,6].reshape(-1,1)).ravel()
    X_val_standardized[:,6] = ss_utau_120_60.transform(X_val[:,6].reshape(-1,1)).ravel()

    joblib.dump(ss_utau_120_60, 'ss_utau_120_60_dns_original.pkl')
 
  #  fig, ax = plt.subplots(7,1, figsize = (6,20))
  #  sns.histplot(X_train_standardized[:,0], stat='density', bins=100, kde=True, ax=ax[0])
  #  sns.histplot(X_train_standardized[:,1], stat='density', bins=100, kde=True, ax=ax[1])
  #  sns.histplot(X_train_standardized[:,2], stat='density', bins=100, kde=True, ax=ax[2])
  #  sns.histplot(X_train_standardized[:,3], stat='density', bins=100, kde=True, ax=ax[3])
  #  sns.histplot(X_train_standardized[:,4], stat='density', bins=100, kde=True, ax=ax[4])
  #  sns.histplot(X_train_standardized[:,5], stat='density', bins=100, kde=True, ax=ax[5])
  #  sns.histplot(X_train_standardized[:,6], stat='density', bins=100, kde=True, ax=ax[6])
  #  #ax[0].set_title('tauwall / tauwall mean norm')
  #  #ax[1].set_title('u60 / utau60_mean norm')
  #  #ax[2].set_title('v60 / utau60_mean norm')
  #  #ax[3].set_title('w60 / utau60_mean norm')
  #  #ax[4].set_title('u120 / utau120_mean norm')
  #  #ax[5].set_title('v120 / utau120_mean norm')
  #  #ax[6].set_title('w120 / utau120_mean norm')
  #  plt.tight_layout()
  #  plt.show()
  #  plt.close()
    X_all = np.concatenate((X_train_standardized, X_test_standardized), axis=0)
    X_all = np.concatenate((X_all, X_val_standardized), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    y_all = np.concatenate((y_all, y_val), axis=0)
    
    traindata_norm = np.hstack((y_train.reshape(-1,1),X_train_standardized))
    testdata_norm = np.hstack((y_test.reshape(-1,1), X_test_standardized))
    valdata_norm = np.hstack((y_val.reshape(-1,1), X_val_standardized))
    alldata_norm = np.hstack((y_all.reshape(-1,1), X_all))

    #np.save('train_dataset_norm_splitted.npy', traindata_norm)
    #np.save('test_dataset_norm_splitted.npy', testdata_norm)
    #np.save('valid_dataset_norm_splitted.npy', valdata_norm)
    #np.save('all_dataset_norm_splitted.npy', alldata_norm)
    
    train_size = int(X_train.shape[0])
    test_size = int(X_test.shape[0])
    valid_size = int(X_val.shape[0])
    all_size = int(X_all.shape[0])

hist, bin_edges = np.histogram(y_train, bins=100, density=True)
bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2

from scipy.interpolate import interp1d
freq_interp = interp1d(bin_center, hist, bounds_error=False, fill_value='extrapolate')
if __name__ == "__main__":

    X_all = torch.tensor(X_all, dtype=torch.float32)
    y_all = torch.tensor(y_all, dtype=torch.float32)
    X_train = torch.tensor(X_train_standardized, dtype=torch.float32)
    X_test = torch.tensor(X_test_standardized, dtype=torch.float32)
    X_val = torch.tensor(X_val_standardized, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)
    all_dataset = TensorDataset(X_all, y_all)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, num_workers=num_workers)
    # # 02. model

# In[8]:

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# model
#modifiche per implementare un Learning Rate scheduler per le fluttuazioni dns

class Cfd_mlp(pl.LightningModule):
    def __init__(self, batch_size, learning_rate, node_per_layer, freq_interp):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.freq_interp = freq_interp
        #self.node_per_layer = node_per_layer
        layer = []
        for i in range(int(len(node_per_layer)-1)):
            layer.append(nn.Linear(node_per_layer[i], node_per_layer[i+1]))
            #layer.append(nn.BatchNorm1d(node_per_layer[i+1]))
            layer.append(nn.LeakyReLU())
           # layer.append(nn.Dropout(p=0.05))
        layer.append(nn.Linear(node_per_layer[-1],1))
        self.mlp = nn.Sequential(*layer)  #unpack the layer
#           nn.Linear(6, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#           nn.ReLU(),
#           nn.Linear(2048, 1),
#       )

        #apply He initialization (works well with ReLU)
        self._init_weights()

    def forward(self, x):
        output = self.mlp(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
       # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-4, max_lr=5e-3, step_size_up=2000, mode='triangular',cycle_momentum=False)
       # scheduler = {
       #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'),
       #     'monitor': 'val_loss',  # nome del log della val loss
       #     'interval': 'epoch',
       #     'frequency': 1
       # }
       # return {'optimizer': optimizer, 'lr_scheduler': scheduler}
       #scheduler = torch.optim.lr_scheduler.OneCycleLR(
       #         optimizer,
       #         max_lr = 1e-3,
       #         steps_per_epoch = len(train_loader),
       #         epochs = 180,
       #         pct_start = 0.25,
       #         anneal_strategy='cos',
       #         final_div_factor=1e4
       #         )

        return optimizer

    def _init_weights(self):
        #Apply He initialization to each layer of my model
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    
    def loss_weights(self,y):
        y = y.unsqueeze(1)
        freqs = torch.tensor(freq_interp(y.cpu().numpy()), dtype=torch.float32).to(device)
        weight = 1.0 / (freqs + 1e-10)
        return weight

    def quantile_loss(self, y, y_hat, tau):
        y = y.unsqueeze(1)
        errore = y - y_hat
        loss_function = torch.mean(torch.maximum(tau * errore, (tau - 1) * errore))
        return loss_function

    def log_cosh_loss(self, y, y_hat):
        y = y.unsqueeze(1)
        arg = torch.mean(torch.cosh(y_hat - y))
        loss_function = torch.mean(torch.log(arg))
        return loss_function

    def MSE_loss(self, y, y_hat):
        y=y.unsqueeze(1)
        loss_function = F.l1_loss(y_hat, y)                             #cambio un attimo mse con mae
        return loss_function

    def log_loss(self, y, y_hat):
        y = y.unsqueeze(1)
        loss =  torch.log(1 + torch.abs(y_hat - y))
        loss = torch.mean(loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.mlp(x)
        loss = self.MSE_loss(y, y_hat) #* self.loss_weights(y)
       # weight = self.loss_weights(y)
       # loss_01 = self.quantile_loss(y, y_hat, 0.1)
       # loss_05 = self.quantile_loss(y, y_hat, 0.5)
       # loss_09 = self.quantile_loss(y, y_hat, 0.9)
      #  loss = torch.mean(loss)
       # loss = loss * weight
       # loss = torch.mean(loss)
       # loss = self.log_cosh_loss(y, y_hat)
       # loss_01 = self.loss_function(y, y_hat, 0.1)
       # loss_05 = self.loss_function(y, y_hat, 0.5)
       # loss_09 = self.loss_function(y, y_hat, 0.9)
       # loss = loss_01 + loss_05 + loss_09
        self.log('train_loss', loss, prog_bar = True, on_epoch = True)
        prog_bar=True
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.mlp(x)
        loss = self.MSE_loss(y, y_hat) 
       # weight = self.loss_weights(y)
       # loss_01 = self.quantile_loss(y, y_hat, 0.1)
       # loss_05 = self.quantile_loss(y, y_hat, 0.5)
       # loss_09 = self.quantile_loss(y, y_hat, 0.9)
       # loss = torch.mean(loss)
       # loss = loss * weight
       # loss = torch.mean(loss)
       # loss = self.log_cosh_loss(y, y_hat)
       # loss_01 = self.loss_function(y, y_hat, 0.1)
       # loss_05 = self.loss_function(y, y_hat, 0.5)
       # loss_09 = self.loss_function(y, y_hat, 0.9)
       # loss = loss_01 + loss_05 + loss_09
        self.log('val_loss', loss, prog_bar=True, on_epoch = True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.mlp(x)
        loss = self.MSE_loss(y, y_hat) #* self.loss_weights(y)
       # weight = self.loss_weights(y)
       # loss_01 = self.quantile_loss(y, y_hat, 0.1)
       # loss_05 = self.quantile_loss(y, y_hat, 0.5)
       # loss_09 = self.quantile_loss(y, y_hat, 0.9)
     #   loss = torch.mean(loss)
       # loss = loss * weight
       # loss = torch.mean(loss)
       # loss = self.log_cosh_loss(y, y_hat)
       # loss_01 = self.loss_function(y, y_hat, 0.1)
       # loss_05 = self.loss_function(y, y_hat, 0.5)
       # loss_09 = self.loss_function(y, y_hat, 0.9)
       # loss = loss_01 + loss_05 + loss_09
        self.log('test_loss', loss, prog_bar=True)

    def train_dataloader(self):
        return DataLoader(train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=num_workers)

    def test_dataloader(self):
        return DataLoader(test_dataset,
                          batch_size=self.batch_size,
                          num_workers=num_workers)
        
    #def per loggare metriche in TensorBoard
   # def validation_epoch_end(self, outputs):
   #     y_true = torch.cat([o['y_true'] for o in outputs]).cpu().numpy()
   #     y_pred = torch.cat([o['y_pred'] for o in outputs]).cpu().numpy()
   #     
   #     mae = mean_absolute_error(y_true, y_pred)
   #     rmse = mean_squared_error(y_true, y_pred, squared=False)
   #     r2 = r2_score(y_true, y_pred)

   #     self.log("val/mae", mae)
   #     self.log("val/rmse", rmse)
   #     self.log("val/r2", r2)
   #     
   #     self.y_true_epoch = y_true
   #     self.y_pred_epoch = y_pred
# # 03. Train

# In[9]:

if __name__ == "__main__":
    # model
    model = Cfd_mlp(batch_size, 1e-4,[7, 1024,512,128], freq_interp = freq_interp)
    
    
    # In[10]:
    
    
    pl.Trainer.__init__
    
    # In[11]:
    
    
    # trainer
    logger = TensorBoardLogger('tb_logs_utau60flut', name='utauall_model_allFEATURES_splitted', version='EWM_enhanced_utau60flut_nondim_targetnorm_maeloss_dns_120_60_correct') #logger per monitoraggio allenamento
    checkpoint = ModelCheckpoint(
        dirpath="checkpoints_utau60flut",
        filename="best_model_EWM_utau60flut_nondim_MSE_targetnorm_dns_120_60_correct-{epoch:02d}-{val_loss:.2f}_nondimtarget",
        monitor='val_loss', 
        save_top_k=3, 
        mode='min') #permette di salvare lo stato del modello e riprendere l'allenamento i  seguito
    earlystop = EarlyStopping(monitor='val_loss', patience=60, mode='min')
    trainer = pl.Trainer(accelerator="gpu",
                         devices=1,
                         #gpu=1,
                         max_epochs=1000,
                         logger=logger,
                         deterministic=True,
                         callbacks=[checkpoint,earlystop],
                         log_every_n_steps=1,
                         #flush_logs_every_n_steps=2,
                         #auto_lr_find=True,
                         # overfit_batches=10,
                        )
    
    
    # In[12]:
    
    
    # # tune to find the best learning rate
    # lr_finder = trainer.tuner.lr_find(model)
    
    # # plot lr_finder
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    
    # # set lr
    # model.learning_rate = lr_finder.suggestion()
    # print("Learning rate:", model.learning_rate)
    
    
    # In[13]:
    #utau_mean  = 4.97576926E-002
    #Retau_mean = 1.02003270E+003
    #delta_v = 0.5 / Retau_mean
    #rho = 1.
    #mu = utau_mean * delta_v
    #u_mean = np.mean(data[:,0])
    #grad_wall_mean = u_mean / utau_mean
    
    # plot results before training
    #from matplotlib.colors import LogNorm
    #test_loader = DataLoader(test_dataset, batch_size=test_size, num_workers=num_workers)
    #x, y_normalized = next(iter(test_loader))
    #y_hat_normalized = model(x)
    #y_hat_normalized = y_hat_normalized.detach().numpy().reshape(-1,1)
    #y_normalized = y_normalized.detach().numpy().reshape(-1,1)
    #y = np.exp(y_normalized.ravel()) - epsilon
    #y_hat = np.exp(y_hat_normalized.ravel()) -epsilon
    #fig,ax = plt.subplots() 
    #HH, xe, ye = np.histogram2d(y.flatten(), y_hat.flatten(), bins = 20, density = True)
    #grid = HH.transpose()
    #midpoints = (xe[1:] + xe[:-1])/2, (ye[1:] + ye[:-1])/2
    #lev = np.linspace(-1e-6,np.max(grid),15)
    #print('MAX', np.max(grid))
    #cs = ax.contourf(*midpoints, grid,levels=lev,cmap='Blues',norm=LogNorm())
    #ax.contour(*midpo#ints, grid,levels=lev,colors='black',linewidths=0.1,norm = LogNorm())
    ##ax.scatter(tauw_dns.flatten()/tauw_mean, tauw_1[:,:int(z1.shape[0])].flatten()/tauw_mean, s=4, label = 'tauwall model vs dns')
    #ax.plot(y.flatten(),y.flatten(), c='k')
    #ax.set_title('model prediction')
    #ax.set_xlabel('tauw_dns/tauw_mean')
    #ax.set_ylabel('tauw_model/tauw_mean')
    #ax.set_ylim(np.min(y.flatten()), np.max(y_hat.flatten()))
    #ax.set_xlim(np.min(y.flatten()), np.max(y.flatten()))
    #cbar = plt.colorbar(cs,ax=ax)
    #cbar.set_label("PDF density")
    ##plt.legend(loc='upper right')
    #plt.grid()
    #plt.savefig('tawall_slice_yplus_60_joint.png', dpi = 300)
    #plt.show()
    #plt.close()
    
    #ax.scatter(y, y_hat,s=4)
    #ax.plot([1.5, 4.5], [1.5, 4.5], c='k')
    #plt.xlabel("Real target")
    #plt.ylabel("Predicted target")
    #plt.show()
    
    
    #print(y.shape)
    #print(test_size)
    #mse_b = mean_squared_error(y ,y_hat )
    #r2_b = r2_score(y ,y_hat)
    #rms_b = np.sqrt(mse_b)
    #print('prestazioni modello before training Retau1000')
    #print('MSE : {}'.format(mse_b))
    #print('MAE : {}'.format(mean_absolute_error(y ,y_hat)))
    #print('R2 : {}'.format(r2_b))
    #print('RMS : {}'.format(rms_b))
    
    # In[14]:
    
    
    #visualizza plot risultati
    #fig, ax = plt.subplots(1,2,figsize=(15,5))
    #sns.scatterplot(x=y , y=y_hat, s=4, color='blue', ax=ax[0])
    #sns.kdeplot(x= y , y= y_hat, cmap='viridis', fill=True,  levels=15, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
    #norm = plt.Normalize(vmin=0, vmax=10)  # valori arbitrari: personalizza!
    #sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    #sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax[1])
    #cbar.set_label("Densità stimata")
    #plt.tight_layout()
    #ax[1].plot(y , y , color='red', linewidth=1)
    ## Inserisci il testo in alto a destra
    #ax[1].text(
    #    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    #    "MSE: {:.6f}\nR2 : {:.6f}\nRms : {:.6f}".format(mse_b, r2_b, rms_b),     # testo da inserire
    #    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    #    fontsize=10,
    #    verticalalignment='top',
    #    horizontalalignment='right',
    #    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    #)
    #ax[0].plot(y , y , color='red', linewidth=1)
    #ax[0].set_xlabel('target')
    #ax[0].set_ylabel('prediction')
    #ax[0].set_title('Risultati before training originale')
    #ax[1].set_xlabel('target')
    #ax[1].set_ylabel('prediction')
    #ax[1].set_title('Risultati before training dataset originale')
    ##plt.show()
    #plt.savefig('Retau1000_MEAN_SUBSAMPLE_before_training_utauall_inputnorm.png', dpi=300)
    #plt.close()
    
    
    # In[15]:
    
    
    #fig,ax = plt.subplots(2,1,figsize=(5,7))
    #sns.histplot(data=y_hat , stat='percent', color='r',bins=50, kde=True, ax = ax[0])
    #sns.histplot(data=y , stat='percent', bins=50, kde=True, ax = ax[1])
    #ax[0].set_xlim(y.min(), y.max())
    #ax[1].set_xlim(y.min(), y.max())
    #ax[0].set_title('prevision')
    #ax[1].set_title('real value')
    #plt.tight_layout()
    ##plt.show()
    #plt.savefig('Retau1000_MEAN_SUBSAMPLE_distribution_before_training_utauall_inputnorm.png', dpi=300)
    #plt.close()
    
    
    # In[ ]:
    
    
    
    
    
    
    
    # In[18]:
    
    
    # train the model
    trainer.fit(model, train_loader, valid_loader)
    
    
    # # 04. Test
    
    # In[20]:
    
    
#trainer.test()
#
#
## In[21]:
#
#
#test_loader = DataLoader(test_dataset, batch_size=test_size, num_workers=num_workers)
#x, y_normalized = next(iter(test_loader))
#y_hat_normalized = model(x)
#y_hat_normalized = y_hat_normalized.detach().numpy().reshape(-1,1)
#y_normalized = y_normalized.detach().numpy().reshape(-1,1)
#y = pt_target.inverse_transform(y_normalized.reshape(-1,1)).ravel()
#y_hat = pt_target.inverse_transform(y_hat_normalized.reshape(-1,1)).ravel()
#y_e_yhat = np.hstack((y.reshape(-1,1), y_hat.reshape(-1,1)))
#np.save('test_prediction_unico_batch_splitted.npy', y_e_yhat)
##y = np.exp(y_normalized.ravel()) - epsilon 
##y_hat = np.exp(y_hat_normalized.ravel()) - epsilon
##fig,ax = plt.subplots() 
##HH, xe, ye = np.histogram2d(y.flatten(), y_hat.flatten(), bins = 10, density = True)
##grid = HH.transpose()
##midpoints = (xe[1:] + xe[:-1])/2, (ye[1:] + ye[:-1])/2
##lev = np.linspace(-1e-6,np.max(grid),40)
##print('MAX', np.max(grid))
##cs = ax.contourf(*midpoints, grid,levels=lev,cmap='Blues',norm=LogNorm())
##ax.contour(*midpoints, grid,levels=lev,colors='black',linewidths=0.1,norm = LogNorm())
###ax.scatter(tauw_dns.flatten()/tauw_mean, tauw_1[:,:int(z1.shape[0])].flatten()/tauw_mean, s=4, label = 'tauwall model vs dns')
##ax.plot(y.flatten(),y.flatten(), c='k')
##ax.set_title('model prediction')
##ax.set_xlabel('tauw_dns/tauw_mean')
##ax.set_ylabel('tauw_model/tauw_mean')
##ax.set_ylim(np.min(y.flatten()), np.max(y.flatten()))
##ax.set_xlim(np.min(y.flatten()), np.max(y.flatten()))
##cbar = plt.colorbar(cs,ax=ax)
##cbar.set_label("PDF density")
###plt.legend(loc='upper right')
##plt.grid()
##plt.savefig('tawall_slice_yplus_60_joint.png', dpi = 300)
##plt.show()
##plt.close()
##x.scatter(y, y_hat, s=4)
##ax.plot([1.5, 4.5], [1.5, 4.5], c='k')
##plt.xlabel("Real target")
##plt.ylabel("Predicted target")
##plt.show()
#
#
## In[22]:
#
#print(y.shape)
#print(test_size)
#mse = mean_squared_error(y , y_hat )
#r2 = r2_score(y , y_hat )
#rms = np.sqrt(mse)
#print('prestazioni modello after training Retau1000 test_dataset in un unico batch')
#print('MSE : {}'.format(mse))
#print('MAE : {}'.format(mean_absolute_error(y ,y_hat)))
#print('R2 : {}'.format(r2))
#print('RMS : {}'.format(rms))
#
##visualizza plot risultati
##fig, ax = plt.subplots(1,2,figsize=(15,5))
##sns.scatterplot(x=y , y=y_hat, s=4, color='blue', ax=ax[0])
##sns.kdeplot(x= y , y= y_hat, cmap='viridis', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
##norm = plt.Normalize(vmin=0, vmax=10)  # valori arbitrari: personalizza!
##sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
##sm.set_array([])
##cbar = plt.colorbar(sm, ax=ax[1])
##cbar.set_label("Densità stimata")
##plt.tight_layout()
##ax[1].plot(y , y , color='red', linewidth=1)
##ax[1].text(
##    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
##    "MSE: {:.6f}\nR2 : {:.6f}\nRMS : {:.6f}".format(mse, r2, rms),     # testo da inserire
##    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
##    fontsize=10,
##    verticalalignment='top',
##    horizontalalignment='right',
##    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
##)
##ax[0].plot(y , y , color='red', linewidth=1)
##ax[0].set_xlabel('target')
##ax[0].set_ylabel('prediction')
##ax[0].set_title('Risultati NN retau1000')
##ax[1].set_xlabel('target')
##ax[1].set_ylabel('prediction')
##ax[1].set_title('Risultati NN retau1000 test_dataset unico batch')
##plt.show()
##plt.savefig('Retau1000_MEAN_SUBSAMPLE_after_training_utauall_inputnorm.png', dpi=300)
##plt.close()
#
#
## In[23]:
#
#
#import seaborn as sns
##fig,ax = plt.subplots(2,1,figsize=(5,7))
##sns.histplot(data=y_hat, stat='density', color='r',bins=50, kde=True, ax = ax[0])
##sns.histplot(data=y, stat='density', bins=50, kde=True, ax = ax[1])
##ax[0].set_xlim(y.min(), y.max())
##ax[1].set_xlim(y.min(), y.max())
##ax[0].set_title('prevision')
##ax[1].set_title('real value')
##plt.tight_layout()
##plt.show()
##plt.savefig('Retau1000_MEAN_SUBSAMPLE_distr_after_training_utauall_inputnorm.png', dpi=300)
##plt.close()
#
##fig, ax = plt.subplots()
##residuals = y - y_hat
##plt.scatter(y_hat, residuals, alpha=0.5)
##plt.axhline(y=0, color='r', linestyle='--')
##plt.xlabel('Valori Predetti')
##plt.ylabel('Residui')
##plt.title('Residual plot')
##plt.grid()
##plt.show()
##plt.savefig('residui_tauwalltarget.png', dpi=300)
##plt.close()
## In[24]:
#
#exit()
## In[31]:
#test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
#all_pred = []
#all_target = []
#with torch.no_grad():
#    for x_normalized, y_normalized in test_loader:
#        y_hat_normalized = model(x)
#        y_hat_normalized = y_hat_normalized.detach().numpy().reshape(-1,1)
#        y_normalized = y_normalized.detach().numpy().reshape(-1,1)
#        y = pt_target.inverse_transform(y_normalized.reshape(-1,1)).ravel()
#        y_hat = pt_target.inverse_transform(y_hat_normalized.reshape(-1,1)).ravel()
#all_pred.append(y_hat)
#all_target(y)
#
#y_e_yhat = np.hstack((all_target.reshape(-1,1), all_pred.reshape(-1,1)))
#np.save('test_prediction_più_batch_splitted.npy', y_e_yhat)
#mse = mean_squared_error(all_target , all_pred )
#r2 = r2_score(all_target , all_pred )
#rms = np.sqrt(mse)
#print('prestazioni modello after training Retau1000 test_dataset in più  batch')
#print('MSE : {}'.format(mse))
#print('MAE : {}'.format(mean_absolute_error(all_target ,all_pred)))
#print('R2 : {}'.format(r2))
#print('RMS : {}'.format(rms))
#
##visualizza plot risultati
##fig, ax = plt.subplots(1,2,figsize=(15,5))
##sns.scatterplot(x=all_target , y=all_pred, s=4, color='blue', ax=ax[0])
##sns.kdeplot(x= all_target , y= all_pred, cmap='viridis', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
##norm = plt.Normalize(vmin=0, vmax=10)  # valori arbitrari: personalizza!
##sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
##sm.set_array([])
##cbar = plt.colorbar(sm, ax=ax[1])
##cbar.set_label("Densità stimata")
##plt.tight_layout()
##ax[1].plot(all_target , all_pred , color='red', linewidth=1)
##ax[1].text(
##    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
##    "MSE: {:.6f}\nR2 : {:.6f}\nRMS : {:.6f}".format(mse, r2, rms),     # testo da inserire
##    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
##    fontsize=10,
##    verticalalignment='top',
##    horizontalalignment='right',
##    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
##)
##ax[0].plot(all_target , all_target , color='red', linewidth=1)
##ax[0].set_xlabel('target')
##ax[0].set_ylabel('prediction')
##ax[0].set_title('Risultati NN retau1000')
##ax[1].set_xlabel('target')
##ax[1].set_ylabel('prediction')
##ax[1].set_title('Risultati NN retau1000 test_dataset più batch')
##plt.show()
##plt.savefig('Retau1000_MEAN_SUBSAMPLE_after_training_utauall_inputnorm_piùbatch.png', dpi=300)
##plt.close()
##
##
##fig,ax = plt.subplots(2,1,figsize=(5,7))
##sns.histplot(data=all_pred, stat='density', color='r',bins=50, kde=True, ax = ax[0])
##sns.histplot(data=all_target, stat='density', bins=50, kde=True, ax = ax[1])
###ax[0].set_xlim(y.min(), y.max())
###ax[1].set_xlim(y.min(), y.max())
##ax[0].set_title('prevision')
##ax[1].set_title('real value')
##plt.tight_layout()
##plt.show()
##plt.savefig('Retau1000_MEAN_SUBSAMPLE_distr_after_training_utauall_inputnorm.png', dpi=300)
##plt.close()
#
## In[24]:
#
#all_loader = DataLoader(all_dataset, batch_size=batch_size, num_workers=num_workers)
#all_pred = []
#all_target = []
#with torch.no_grad():
#    for x_normalized, y_normalized in all_loader:
#        y_hat_normalized = model(x)
#        y_hat_normalized = y_hat_normalized.detach().numpy().reshape(-1,1)
#        y_normalized = y_normalized.detach().numpy().reshape(-1,1)
#        y = pt_target.inverse_transform(y_normalized.reshape(-1,1)).ravel()
#        y_hat = pt_target.inverse_transform(y_hat_normalized.reshape(-1,1)).ravel()
#all_pred.append(y_hat)
#all_target(y)
#
#y_e_yhat = np.hstack((all_pred.reshape(-1,1), all_target.reshape(-1,1)))
#np.save('alldatatset_prediction_piu_batch_splitted.npy', y_e_yhat)
#mse = mean_squared_error(all_target , all_pred )
#r2 = r2_score(all_target , all_pred )
#rms = np.sqrt(mse)
#print('prestazioni modello after training Retau1000 all_datset')
#print('MSE : {}'.format(mse))
#print('MAE : {}'.format(mean_absolute_error(all_target ,all_pred)))
#print('R2 : {}'.format(r2))
#print('RMS : {}'.format(rms))
#
##visualizza plot risultati
##fig, ax = plt.subplots(1,2,figsize=(15,5))
##sns.scatterplot(x=all_target , y=all_pred, s=4, color='blue', ax=ax[0])
##sns.kdeplot(x= all_target , y= all_pred, cmap='viridis', fill=True,  levels=30, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
##norm = plt.Normalize(vmin=0, vmax=10)  # valori arbitrari: personalizza!
##sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
##sm.set_array([])
##cbar = plt.colorbar(sm, ax=ax[1])
##cbar.set_label("Densità stimata")
##plt.tight_layout()
##ax[1].plot(all_target , all_pred , color='red', linewidth=1)
##ax[1].text(
##    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
##    "MSE: {:.6f}\nR2 : {:.6f}\nRMS : {:.6f}".format(mse, r2, rms),     # testo da inserire
##    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
##    fontsize=10,
##    verticalalignment='top',
##    horizontalalignment='right',
##    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
##)
##ax[0].plot(all_target , all_target , color='red', linewidth=1)
##ax[0].set_xlabel('target')
##ax[0].set_ylabel('prediction')
##ax[0].set_title('Risultati NN retau1000')
##ax[1].set_xlabel('target')
##ax[1].set_ylabel('prediction')
##ax[1].set_title('Risultati NN retau1000 all dataset')
##plt.show()
##plt.savefig('Retau1000_MEAN_SUBSAMPLE_after_training_utauall_inputnorm_alldataset.png', dpi=300)
##plt.close()
##
##
##fig,ax = plt.subplots(2,1,figsize=(5,7))
##sns.histplot(data=all_pred, stat='density', color='r',bins=50, kde=True, ax = ax[0])
##sns.histplot(data=all_target, stat='density', bins=50, kde=True, ax = ax[1])
###ax[0].set_xlim(y.min(), y.max())
###ax[1].set_xlim(y.min(), y.max())
##ax[0].set_title('prevision')
##ax[1].set_title('real value')
##plt.tight_layout()
##plt.show()
##plt.savefig('Retau1000_MEAN_SUBSAMPLE_distr_after_training_utauall_alldataset.png', dpi=300)
##plt.close()
#
## In[31]:
##train_loader = DataLoader(train_dataset, batch_size=train_size, num_workers=num_workers)
##x, y_normalized = next(iter(train_loader))
##y_hat_normalized = model(x)
##y = scaler.inverse_transform(y_normalized.reshape(-1,1).detach().numpy())
##y_hat = scaler.inverse_transform(y_hat_normalized.reshape(-1,1).detach().numpy())
##fig, ax = plt.subplots()
##
##ax.scatter(y, y_hat, s=1)
##ax.plot([1.5, 4.5], [1.5, 4.5], c='k')
##plt.xlabel("Real target")
##plt.ylabel("Predicted target")
##plt.show()
##
##
### In[32]:
##
##
##valid_loader = DataLoader(valid_dataset, batch_size=valid_size, num_workers=num_workers)
##x, y_normalized = next(iter(valid_loader))
##y_hat_normalized = model(x)
##y = scaler.inverse_transform(y_normalized.reshape(-1,1).detach().numpy())
##y_hat = scaler.inverse_transform(y_hat_normalized.reshape(-1,1).detach().numpy())
##fig, ax = plt.subplots()
##
##ax.scatter(y, y_hat, s=4)
##ax.plot([1.5, 4.5], [1.5, 4.5], c='k')
##plt.xlabel("Real target")
##plt.ylabel("Predicted target")
##plt.show()
##
#
## In[ ]:
    



