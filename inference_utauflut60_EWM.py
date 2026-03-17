import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,kurtosis
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.init as init

from Retau1000_utauflut_dimensionalized import Cfd_mlp
from Retau1000_utauflut_dimensionalized import freq_interp
import joblib
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Cfd_mlp.load_from_checkpoint("/home/dalmonte/CFD/MLWM/cfd_ml_update/notebooks/02_lighting/Retau_1000/tauwall_target/checkpoints_utau60flut/best_model_EWM_utau60flut_nondim_MSE_targetnorm_dns_120_60_correct-epoch=69-val_loss=0.77_nondimtarget.ckpt",
                                     batch_size = 64,
                                     learning_rate = 1e-4,
                                     node_per_layer = [7,1024, 512,128],
                                     freq_interp = freq_interp
                                     )
model.eval()
model.to("cuda")

#utau10_flut_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/utau60_flut_test_nondim_dns.npz'))      #questo è quello dns
u10_test = dict(np.load('u10_test.npz'))      #questo è quello dns
v10_test = dict(np.load('v10_test.npz'))      #questo è quello dns
w10_test = dict(np.load('w10_test.npz'))      #questo è quello dns
tauwall_test = dict(np.load('tauwall_test.npz'))
#u30_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/u120_test_utau120EWM.npz'))
#v30_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/v120_test_utau120EWM.npz'))
#w30_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/w120_test_utau120EWM.npz'))
#u10_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/u60_test_utau60EWM.npz'))
#v10_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/v60_test_utau60EWM.npz'))
#w10_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/w60_test_utau60EWM.npz'))
#utau_30_10_test = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/utau120_utau60_EWM_test.npz'))
#utau10_EWM = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/utau60_EWM_test.npz'))
#utau30_EWM = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/utau120_EWM_test.npz'))
ss_u10 = joblib.load('ss_u10.pkl') 
ss_v10 = joblib.load('ss_v10.pkl') 
ss_w10 = joblib.load('ss_w10.pkl') 
#ss_u10 = joblib.load('ss_u30_utau30_dns.pkl') 
#ss_v10 = joblib.load('ss_v30_utau30_dns.pkl') 
#ss_w10 = joblib.load('ss_w30_utau30_dns.pkl') 
#ss_utau_30_10 = joblib.load('ss_utau_120_60_dns_original.pkl') 
ss_target = joblib.load('ss_targets.pkl')                #questo è quello dns
#mm_u60 = joblib.load('mm_u60_splitted.pkl') 
#mm_v60 = joblib.load('mm_v60_splitted.pkl') 
#mm_w60 = joblib.load('mm_w60_splitted.pkl') 
#mm_u120 = joblib.load('mm_u120_splitted.pkl') 
#mm_v120 = joblib.load('mm_v120_splitted.pkl') 
#mm_w120 = joblib.load('mm_w120_splitted.pkl')

#pt_target = joblib.load('pt_target_splitted.pkl')


prediction_test = {}
prediction_test_norm = {}
for (key, chiaveu10, chiavev10, chiavew10) in zip(tauwall_test.keys(), u10_test.keys(), v10_test.keys(), w10_test.keys())#, u10_test.keys(), v10_test.keys(), w10_test.keys()):

    #u30 = u30_test[chiaveu30].ravel()
    #v30 = v30_test[chiavev30].ravel()
    #w30 = w30_test[chiavew30].ravel()
    u10 = u10_test[chiaveu10].ravel()
    v10 = v10_test[chiavev10].ravel()
    w10 = w10_test[chiavew10].ravel()
    tauwall = tauwall_test[key].ravel()
    #utau_120_60 = np.log1p(utau_120_60)
    #u30 = ss_u30.transform(u30.reshape(-1,1)).ravel()
    #v30 = ss_v30.transform(v30.reshape(-1,1)).ravel()
    #w30 = ss_w30.transform(w30.reshape(-1,1)).ravel()
    u10 = ss_u10.transform(u10.reshape(-1,1)).ravel()
    v10 = ss_v10.transform(v10.reshape(-1,1)).ravel()
    w10 = ss_w10.transform(w10.reshape(-1,1)).ravel()
    tauwall = ss_target.transform(tauwall.reshape(-1,1)).ravel()
   # u60 = mm_u60.transform(u60.reshape(-1,1)).ravel()
   # v60 = mm_v60.transform(v60.reshape(-1,1)).ravel()
   # w60 = mm_w60.transform(w60.reshape(-1,1)).ravel()
   # u120 = mm_u120.transform(u120.reshape(-1,1)).ravel()
   # v120 = mm_v120.transform(v120.reshape(-1,1)).ravel()
   # w120 = mm_w120.transform(w120.reshape(-1,1)).ravel()

    X_test = np.zeros((u10.shape[0], 3))
    X_test[:,0] = u10
    X_test[:,1] = v10
    X_test[:,2] = w10

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad(): 
        pred = model(X_test)

        pred = pred.detach().cpu().numpy()
        #pred = np.expm1(pred)
        pred = ss_target.inverse_transform(pred.reshape(-1,1))
        #pred = np.exp(pred) -1
        pred = pred.reshape(121,64)
        prediction_test[key] = pred


Retau_mean = 1.02003270E+003
utau_mean  = 4.97576926E-002
lz = 1
delta  = lz / 2
delta_v = delta / Retau_mean
rho = 1.
mu = utau_mean * delta_v
tauwall_stat = rho * (utau_mean**2)
ywall = 0.0001442222

#print(utau60_EWM['u_60_49'])
#print(prediction_test['u_60_49'])

#for keys in utau60_EWM.keys():
#    prediction_test[key] = prediction_test[key] * utau60_EWM[key]

#print(prediction_test['u_60_49'])

#utau10_flut_dns = {}
#for key in prediction_test_nondim.keys():
#    campo = np.zeros((121,64))
#    campo_pred = utau10_flut_test[key]
#    campo_utau10 = utau10_EWM[key]
#    for i in range(utau10_EWM['u_60_49'].shape[0]):
#        for j in range(utau10_EWM['u_60_49'].shape[1]):
#            campo[i,j] = campo_pred[i,j] * campo_utau10[i,j]
#    utau10_flut_dns[key] = campo
#print('-------------------------------------------------------------------------------------------------------')    
#print('-------------------------------------------------------------------------------------------------------')    
#prediction_test = {}
#for key in prediction_test_nondim.keys():
#    campo = np.zeros((121,64))
#    campo_pred = prediction_test_nondim[key]
#    campo_utau10 = utau10_EWM[key]
#    for i in range(utau10_EWM['u_60_49'].shape[0]):
#        for j in range(utau10_EWM['u_60_49'].shape[1]):
#            campo[i,j] = campo_pred[i,j] * campo_utau10[i,j]
#    prediction_test[key] = campo        
#        
                
#print(utau10_flut_dns['u_60_49'])
#
#print('-------------------------------------------------------------------------------------------------------')   
#print('prediction')
#print(prediction_test['u_60_49'])
#print('-------------------------------------------------------------------------------------------------------')   
#print('errore')
#print(np.abs(utau10_flut_dns['u_60_49'] - prediction_test['u_60_49']))
#print('-------------------------------------------------------------------------------------------------------')    
#print(np.mean(np.abs(utau10_flut_dns['u_60_49'] - prediction_test['u_60_49'])))

lx = 5.
ly = 1.5
nx = 1216
ny = 640

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

x_less = x[::10]
y_less = y[::10]

pred_errore = {}
for (key,keys) in zip(tauwall_test.keys(),prediction_test.keys()):
    pred_errore[key] = (prediction_test[keys] - tauwall_test[key]) / tauwall_test[key]
    fig,ax = plt.subplots(3,1,figsize=(12,20))#, constrained_layout=True)
    utaumean_pred = np.mean(prediction_test[key], axis=(0,1))
    utaumean_test = np.mean(tauwall_test[key], axis=(0,1))
    fig.suptitle('tauwall_test_mean = {:.5f}, tauwall_pred_mean = {:.5f}'.format(utaumean_test, utaumean_pred))
    ax[0].axis('scaled')
    ax[1].axis('scaled')
    ax[2].axis('scaled')
    vmin = utau10_flut_dns[key].min()
    vmax = utau10_flut_dns[key].max()
    vmin_errore = -1
    vmax_errore = 1
    s1 = ax[0].pcolormesh(x_less[:-1],y_less,tauwall_test[key].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
    s2 = ax[1].pcolormesh(x_less[:-1],y_less,prediction_test[keys].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
    e1 = ax[2].pcolormesh(x_less[:-1],y_less,pred_errore[key].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
    ax[0].set_xlim(0,5)
    ax[0].set_ylim(0,1.5)
    ax[1].set_xlim(0,5)
    ax[1].set_ylim(0,1.5)
    ax[2].set_xlim(0,5)
    ax[2].set_ylim(0,1.5)
    #fig.colorbar(s1, ax = [ax[0], ax[1], ax[2]], orientation='vertical', shrink=0.8, pad=1)
    #plt.tight_layout()
    ax[0].set_title('slice target {}'.format(key))
    ax[1].set_title('slice predict {}'.format(keys))
    ax[2].set_title('errore {}'.format(key))
    # Posizione manuale della colorbar (x0, y0, width, height)
    cbar_ax = fig.add_axes([0.91, 0.37, 0.01, 0.5])
    cbar_ax1 = fig.add_axes([0.91, 0.03, 0.01, 0.3])
    fig.colorbar(s1, cax=cbar_ax, label='Valore')
    fig.colorbar(e1, cax=cbar_ax1, label='Errore')
    plt.subplots_adjust(right=0.9)  # lascia spazio alla colorbar
    plt.savefig('RESULTS/Slice_targetnorm_{}.png'.format(key), dpi=300)
    #plt.show()
    plt.close()

y_hat = np.array([])
y = np.array([])
for key in prediction_test.keys():
    campo_pred = prediction_test[key].ravel()
    campo_target = tauwall_test[key].ravel()
    y_hat = np.append(y_hat, campo_pred)
    y = np.append(y, campo_target)

residual = y_hat + y
print('pred + target mean = {}'.format(np.mean(residual))) 
residual = y_hat - y
print('pred - target mean = {}'.format(np.mean(residual))) 
mse = mean_squared_error(y , y_hat )
r2 = r2_score(y , y_hat )
r2_inverso = r2_score(-y, y_hat)
rms = np.sqrt(mse)
mae = mean_absolute_error(y , y_hat)
print('prestazioni modello Retau1000 utau10 LeakyReLU, MSE')
print('MSE : {}'.format(mse))
print('MAE : {}'.format(mean_absolute_error(y , y_hat)))
print('R2 : {}'.format(r2))
print('R2_inverso : {}'.format(r2_inverso))
print('RMS : {}'.format(rms))
print('target pred corr = {}'.format(np.corrcoef(y, y_hat)[0,1]))
print('target pred corr inversa= {}'.format(np.corrcoef(-y, y_hat)[0,1]))
errore = np.abs(y - y_hat)
print('errore minimo = {}'.format(errore.min()))
print('errore massimo = {}'.format(errore.max()))
print('85esimo percentile test_set = {}'.format(np.percentile(errore, 85)))



fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x= y , y= y_hat, s=4, color='blue', ax=ax[0])
sns.kdeplot(x= y, y= y_hat, cmap='inferno', fill=True, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
norm = plt.Normalize(vmin=0, vmax=5)  # valori arbitrari: personalizza!
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label("Densità stimata")
plt.tight_layout()
ax[1].plot(y, y, color='black', linewidth=1)
ax[1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MSE: {:.7f}\nR2 : {:.7f}\nRMS : {:.7f}\n MAE: {:.7f}".format(mse, r2, rms, mae),     # testo da inserire
    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax[0].plot(y, y, color='black', linewidth=1)
ax[0].set_xlim(y.min(), y.max())
ax[0].set_ylim(y.min(), y.max())
ax[1].set_xlim(y.min(), y.max())
ax[1].set_ylim(y.min(), y.max())
ax[0].set_xlabel('tauwalll target')
ax[0].set_ylabel('tauwall pred ')
ax[0].set_title('tauwall')
ax[1].set_xlabel('tauwall target')
ax[1].set_ylabel('tauwall pred')
ax[1].set_title('tauwall')
plt.savefig('RESULTS/results.png',dpi=300)
plt.show()
#plt.savefig('utau60_targetnorm_nondim_dns.png', dpi=300)
plt.close()

exit()
utau_target = {}
utau_pred = {}

for (key, keypred) in zip(utau10_flut_dns.keys(), prediction_test.keys()):
    utau_target[key] =  utau10_flut_dns[key] + utau10_EWM[keypred]
    utau_pred[keypred] = + prediction_test[keypred] + utau10_EWM[keypred]

np.savez('utau_target_correct.npz', **utau_target)
np.savez('utau60_pred_NN_correct.npz', **utau_pred)
pred_utau_errore = {}
for (key,keys) in zip(utau_target.keys(),utau_pred.keys()):
    pred_utau_errore[key] = (utau_pred[keys] - utau_target[key]) / utau_target[key]
    fig,ax = plt.subplots(3,1,figsize=(12,20))#, constrained_layout=True)
    utaumean_pred = np.mean(utau_pred[key], axis=(0,1))
    utaumean_test = np.mean(utau_target[key], axis=(0,1))
    fig.suptitle('utau_test_mean = {:.5f}, utau_pred_mean = {:.5f}'.format(utaumean_test, utaumean_pred))
    ax[0].axis('scaled')
    ax[1].axis('scaled')
    ax[2].axis('scaled')
    vmin = utau_pred[key].min()
    vmax = utau_pred[key].max()
    vmin_errore = -1
    vmax_errore = 1
    s1 = ax[0].pcolormesh(x_less[:-1],y_less,utau_target[key].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
    s2 = ax[1].pcolormesh(x_less[:-1],y_less,utau_pred[keys].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
    e1 = ax[2].pcolormesh(x_less[:-1],y_less,pred_utau_errore[key].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
    ax[0].set_xlim(0,5)
    ax[0].set_ylim(0,1.5)
    ax[1].set_xlim(0,5)
    ax[1].set_ylim(0,1.5)
    ax[2].set_xlim(0,5)
    ax[2].set_ylim(0,1.5)
    #fig.colorbar(s1, ax = [ax[0], ax[1], ax[2]], orientation='vertical', shrink=0.8, pad=1)
    #plt.tight_layout()
    ax[0].set_title('slice target {}'.format(key))
    ax[1].set_title('slice predict {}'.format(keys))
    ax[2].set_title('errore {}'.format(key))
    # Posizione manuale della colorbar (x0, y0, width, height)
    cbar_ax = fig.add_axes([0.91, 0.37, 0.01, 0.5])
    cbar_ax1 = fig.add_axes([0.91, 0.03, 0.01, 0.3])
    fig.colorbar(s1, cax=cbar_ax, label='Valore')
    fig.colorbar(e1, cax=cbar_ax1, label='Errore')
    plt.subplots_adjust(right=0.9)  # lascia spazio alla colorbar
    plt.savefig('utau60_nondim_dns/utau60_correct/Slice_targetnorm_{}.png'.format(key), dpi=300)      #utau10_correct_inversefluctuation/
    #plt.show()
    plt.close()


#utau_target = dict(np.load('/home/dalmonte/CFD/MLWM/Confronto_modelli/utau_dns_test.npz'))

yu_hat = np.array([])
yu = np.array([])
err_uplus_10 = []
rms_uplus_10 = []
err_uplus_30 = []
rms_uplus_30 = []
for (chiave,key,chiave10,chiave30) in zip(utau_target.keys(),prediction_test.keys(), utau10_EWM.keys(), utau30_EWM.keys()):
    campo_pred = utau_pred[key].ravel()
    campo_target = utau_target[chiave].ravel()
    yu_hat = np.append(yu_hat, campo_pred)
    yu = np.append(yu, campo_target)
    utau = utau_target[chiave].ravel()
    utau_prediction = utau_pred[key].ravel()
    #utau10 = utau10_EWM[chiave10].ravel()
    #utau30 = utau30_EWM[chiave30].ravel()
    u10 = u10_test[chiave10].ravel()
    u30 = u30_test[chiave30].ravel()
    uplus_dns10 = np.mean(u10/utau)
    uplus_dns30 = np.mean(u30/utau)
    uplus_NN10 = np.mean(u10/utau_prediction)
    uplus_NN30 = np.mean(u30/utau_prediction)
    err_u10 = (np.abs(uplus_NN10 - uplus_dns10)) / uplus_dns10
    err_u30 = (np.abs(uplus_NN30 - uplus_dns30)) / uplus_dns30
    err_uplus_10.append(err_u10)
    err_uplus_30.append(err_u30)
    rms_NN10 = np.sqrt(np.mean(np.square(u10/utau_prediction)))
    rms_NN30 = np.sqrt(np.mean(np.square(u30/utau_prediction)))
    rms_dns10 = np.sqrt(np.mean(np.square(u10/utau)))
    rms_dns30 = np.sqrt(np.mean(np.square(u30/utau)))
    rms10 = (np.abs(rms_NN10 - rms_dns10)) / rms_dns10
    rms30 = (np.abs(rms_NN30 - rms_dns30)) / rms_dns30
    rms_uplus_10.append(rms10)
    rms_uplus_30.append(rms30)

err_uplus_10 = np.array(err_uplus_10)
err_uplus_30 = np.array(err_uplus_30)
rms_uplus_10 = np.array(rms_uplus_10)
rms_uplus_30 = np.array(rms_uplus_30)

err_uplus_10  = np.mean(err_uplus_10)
err_uplus_30 = np.mean(err_uplus_30)
rms_uplus_10  = np.mean(rms_uplus_10)
rms_uplus_30 = np.mean(rms_uplus_30)

mseu = mean_squared_error(yu , yu_hat )
r2u = r2_score(yu , yu_hat )
rmsu = np.sqrt(mse)
maeu = mean_absolute_error(yu , yu_hat)
print('prestazioni modello Retau1000 utau60 LeakyReLU, MSE')
print('MSE = {}'.format(mseu))
print('MAE = {}'.format(mean_absolute_error(yu , yu_hat)))
print('R2 = {}'.format(r2u))
print('RMS = {}'.format(rmsu))
print('RE10 = {} %'.format(err_uplus_10 * 100))
print('RMS10 = {} %'.format(rms_uplus_10 * 100))
print('RE30 = {} %'.format(err_uplus_30 * 100))
print('RMS30 = {} %'.format(rms_uplus_30 * 100))

erroreu = np.abs(yu - yu_hat)
print('errore minimo = {}'.format(erroreu.min()))
print('errore massimo = {}'.format(erroreu.max()))
print('85esimo percentile test_set = {}'.format(np.percentile(erroreu, 85)))

fig, ax = plt.subplots(1,2,figsize=(15,5))
sns.scatterplot(x= yu , y= yu_hat, s=4, color='blue', ax=ax[0])
sns.kdeplot(x= yu, y= yu_hat, cmap='inferno', fill=True, thresh=0.01, bw_adjust=1, alpha=0.7, legend=True, ax=ax[1])
norm = plt.Normalize(vmin=0, vmax=5)  # valori arbitrari: personalizza!
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1])
cbar.set_label("Densità stimata")
plt.tight_layout()
ax[1].plot(yu, yu, color='black', linewidth=1)
ax[1].text(
    0.95, 0.95,                 # coordinate relative (95% a destra, 95% in alto)
    "MSE: {:.7f}\nR2 : {:.7f}\nRMS_u10 : {:.7f} %\n RE_u10: {:.7f} %\nRMS_u30 : {:.7f} %\nRE_u30 : {:.7f} %".format(mseu, r2u, rms_uplus_10, err_uplus_10, rms_uplus_30, err_uplus_30),     # testo da inserire
    transform=ax[1].transAxes,  # usa assi normalizzati (da 0 a 1)
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
)
ax[0].plot(yu, yu, color='black', linewidth=1)
ax[0].set_xlim(yu.min(), yu.max())
ax[0].set_ylim(yu.min(), yu.max())
ax[1].set_xlim(yu.min(), yu.max())
ax[1].set_ylim(yu.min(), yu.max())
ax[0].set_xlabel('utau60  target')
ax[0].set_ylabel('utau60  prediction ')
ax[0].set_title('utau60')
ax[1].set_xlabel('utau60  target')
ax[1].set_ylabel('utau60  prediction')
ax[1].set_title('utau60')
plt.show()
#plt.savefig('utau60_targetnorm_nondim_dns.png', dpi=300)
plt.close()

exit()
np.savez('/home/dalmonte/CFD/MLWM/utau60_flut_test_EWM_dns.npz', **utau60_flut_dns)
np.savez('/home/dalmonte/CFD/MLWM/utau60flut_pred_test_EWM_nondim_dns.npz', **prediction_test)
np.savez('/home/dalmonte/CFD/MLWM/errore_utau60flut_test_EWM_nondim_dns.npz', **pred_errore)
exit()
tauwall_train = dict(np.load('/home/dalmonte/CFD/MLWM/tauwall_train_utau_tauwall_stat.npz'))
u60_train = dict(np.load('/home/dalmonte/CFD/MLWM/u60_train_utau_tauwall_stat.npz'))
v60_train = dict(np.load('/home/dalmonte/CFD/MLWM/v60_train_utau_tauwall_stat.npz'))
w60_train = dict(np.load('/home/dalmonte/CFD/MLWM/w60_train_utau_tauwall_stat.npz'))
u120_train = dict(np.load('/home/dalmonte/CFD/MLWM/u120_train_utau_tauwall_stat.npz'))
v120_train = dict(np.load('/home/dalmonte/CFD/MLWM/v120_train_utau_tauwall_stat.npz'))
w120_train = dict(np.load('/home/dalmonte/CFD/MLWM/w120_train_utau_tauwall_stat.npz'))

prediction_train = {}
prediction_train_norm = {}
for (key, chiaveu60, chiavev60, chiavew60, chiaveu120, chiavev120,chiavew120) in zip(tauwall_train.keys(), u60_train.keys(), v60_train.keys(), w60_train.keys(), u120_train.keys(), v120_train.keys(), w120_train.keys()):

    u60 = u60_train[chiaveu60].ravel()
    v60 = v60_train[chiavev60].ravel()
    w60 = w60_train[chiavew60].ravel()
    u120 = u120_train[chiaveu120].ravel()
    v120 = v120_train[chiavev120].ravel()
    w120 = w120_train[chiavew120].ravel()
    u60 = ss_u60.transform(u60.reshape(-1,1)).ravel()
    v60 = ss_v60.transform(v60.reshape(-1,1)).ravel()
    w60 = ss_w60.transform(w60.reshape(-1,1)).ravel()
    u120 = ss_u120.transform(u120.reshape(-1,1)).ravel()
    v120 = ss_v120.transform(v120.reshape(-1,1)).ravel()
    w120 = ss_w120.transform(w120.reshape(-1,1)).ravel()
   # u60 = mm_u60.transform(u60.reshape(-1,1)).ravel()
   # v60 = mm_v60.transform(v60.reshape(-1,1)).ravel()
   # w60 = mm_w60.transform(w60.reshape(-1,1)).ravel()
   # u120 = mm_u120.transform(u120.reshape(-1,1)).ravel()
   # v120 = mm_v120.transform(v120.reshape(-1,1)).ravel()
   # w120 = mm_w120.transform(w120.reshape(-1,1)).ravel()

    X_train = np.zeros((u60.shape[0], 6))
    X_train[:,0] = u60
    X_train[:,1] = v60
    X_train[:,2] = w60
    X_train[:,3] = u120
    X_train[:,4] = v120
    X_train[:,5] = w120

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_norm = model(X_train)

       # pred = pred.detach().cpu().numpy() 
       # pred = np.exp(pred) -1
        pred_norm = pred_norm.detach().cpu().numpy()
        pred = pt_target.inverse_transform(pred_norm.reshape(-1,1))
        pred_norm = pred_norm.reshape(121,64)
        prediction_train_norm[key] = pred_norm
        pred = pred.reshape(121,64)
        prediction_train[key] = pred
#np.savez('/home/dalmonte/CFD/MLWM/tauwall_test_NN_dimensionalized_tauwallstat.npz', **tauwall_test)
#np.savez('/home/dalmonte/CFD/MLWM/prediction_LR_HL_tauwallstat.npz', **prediction_test)
#np.savez('/home/dalmonte/CFD/MLWM/errore_LR_HL_tauwallstat.npz', **pred_errore)

#normalizzo il target
tauwall_train_norm = {}
for key in tauwall_train.keys():
    campo = tauwall_train[key]
    campo = campo.ravel()
    campo = pt_target.transform(campo.reshape(-1,1))
    tauwall_train_norm[key] = campo

for key in tauwall_train.keys():
        tauwall_train[key] = tauwall_train[key] * tauwall_stat
        prediction_train[key] = prediction_train[key] * tauwall_stat

pred_errore_train = {}
for (key,keys) in zip(tauwall_train.keys(),prediction_train.keys()):
    pred_errore_train[key] = (prediction_train[keys] - tauwall_train[key]) / tauwall_stat
    fig,ax = plt.subplots(3,1,figsize=(12,20))#, constrained_layout=True)
    utaumean_pred = np.mean(prediction_train[key], axis=(0,1))
    utaumean_train = np.mean(tauwall_train[key], axis=(0,1))
    fig.suptitle('tauwall_train_mean = {:.5f}, tauwall_pred_mean = {:.5f}'.format(utaumean_train, utaumean_pred))
    ax[0].axis('scaled')
    ax[1].axis('scaled')
    ax[2].axis('scaled')
    vmin = 0.0004#min(tauwall_test[key].min(), prediction_test[keys].min())
    vmax = 0.006#max(tauwall_test[key].max(), prediction_test[keys].max())
    vmin_errore = -1
    vmax_errore = 1
    s1 = ax[0].pcolormesh(x_less[:-1],y_less,tauwall_train[key].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
    s2 = ax[1].pcolormesh(x_less[:-1],y_less,prediction_train[keys].T, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
    e1 = ax[2].pcolormesh(x_less[:-1],y_less,pred_errore_train[key].T, vmin=vmin_errore, vmax=vmax_errore, cmap='coolwarm', shading='auto')
    ax[0].set_xlim(0,5)
    ax[0].set_ylim(0,1.5)
    ax[1].set_xlim(0,5)
    ax[1].set_ylim(0,1.5)
    ax[2].set_xlim(0,5)
    ax[2].set_ylim(0,1.5)
    #fig.colorbar(s1, ax = [ax[0], ax[1], ax[2]], orientation='vertical', shrink=0.8, pad=1)
    #plt.tight_layout()
    ax[0].set_title('slice target {}'.format(key))
    ax[1].set_title('slice predict {}'.format(keys))
    ax[2].set_title('errore {}'.format(key))
    # Posizione manuale della colorbar (x0, y0, width, height)
    cbar_ax = fig.add_axes([0.91, 0.37, 0.01, 0.5])
    cbar_ax1 = fig.add_axes([0.91, 0.03, 0.01, 0.3])
    fig.colorbar(s1, cax=cbar_ax, label='Valore')
    fig.colorbar(e1, cax=cbar_ax1, label='Errore')
    plt.subplots_adjust(right=0.9)  # lascia spazio alla colorbar
    plt.savefig('Slice_leakyReLU_huberloss_tauwallstat/TRAIN_tauwallstat_morelayer_{}.png'.format(key), dpi=300)
    #plt.show()
    plt.close()

mse = mean_squared_error(y , y_hat )
r2 = r2_score(y , y_hat )
rms = np.sqrt(mse)
print('prestazioni modello Retau1000 splitted LeakyReLU, huber_loss, Dropout=0.05')
print('MSE : {}'.format(mse))
print('MAE : {}'.format(mean_absolute_error(y , y_hat)))
print('R2 : {}'.format(r2))
print('RMS : {}'.format(rms))

errore = np.abs(y - y_hat)
print('errore minimo = {}'.format(errore.min()))
print('errore massimo = {}'.format(errore.max()))
print('85esimo percentile test_set = {}'.format(np.percentile(errore, 85)))

y_hat_norm = np.array([])
y_norm = np.array([])
for key in prediction_test_norm.keys():
    campo_pred = prediction_test_norm[key].ravel()
    campo_target = tauwall_test_norm[key].ravel()
    y_hat_norm = np.append(y_hat_norm, campo_pred)
    y_norm = np.append(y_norm, campo_target)

mse = mean_squared_error(y_norm , y_hat_norm )
r2 = r2_score(y_norm , y_hat_norm )
rms = np.sqrt(mse)
print('prestazioni modello Retau1000 splitted LeakyReLU, huber_loss, Dropout=0.05 NORM')
print('MSE : {}'.format(mse))
print('MAE : {}'.format(mean_absolute_error(y_norm , y_hat_norm)))
print('R2 : {}'.format(r2))
print('RMS : {}'.format(rms))

errore_norm = np.abs(y_norm - y_hat_norm)
print('errore minimo = {}'.format(errore_norm.min()))
print('errore massimo = {}'.format(errore_norm.max()))
print('85esimo percentile test_set = {}'.format(np.percentile(errore_norm, 85)))    
