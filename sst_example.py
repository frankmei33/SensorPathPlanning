import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import linalg
import scipy as sp
from scipy.linalg import qr
from netCDF4 import Dataset
from pydmd import DMD, OptDMD
from OOQR_net import OOQR
from grid_net import *

def plot_sst(x,mask):
    snapshot = float("nan")*np.ones((180,360))
    snapshot[mask] = x

    plt.imshow(snapshot, cmap=plt.cm.coolwarm)
    plt.xticks([])
    plt.yticks([])

def plot_traj(Ps,head_width=3,lw=1):
    # plt.figure()
    P = Ps.flatten()
    x,y = grid.ind_to_loc(P)
    x = x.reshape(Ps.shape)
    y = y.reshape(Ps.shape)
    # plt.figure(figsize=(10,5))
    for j in range(Ps.shape[1]):
        for i in range(Ps.shape[0]):
            dx = x[(i+1)%len(x),j] - x[i,j]
            dy = y[(i+1)%len(x),j] - y[i,j]
            plt.arrow(x[i,j],y[i,j],dx,dy,color='yellow',lw=lw,head_width=head_width,length_includes_head=True)
        plt.scatter(x[1:,j],y[1:,j],color='k',zorder=3,s=4*lw**2)
        plt.scatter(x[0,j],y[0,j],color='k',marker='*',zorder=4,s=4*lw**4)
    plot_sst(X[0],mask)
    plt.tight_layout()


#%% read data
f = Dataset('../../Datasets/NOAA_sea_surface_temp/sst.wkmean.1990-present.nc')
lat,lon = f.variables['lat'], f.variables['lon']
SST = f.variables['sst']
sst = SST[:]
f = Dataset('../../Datasets/NOAA_sea_surface_temp/lsmask.nc')
mask = np.squeeze(f.variables['mask']).astype(bool)
X = sst[:,mask]
X = np.reshape(X.compressed(), X.shape) # convert masked array to array
X = X-X.mean() # subtract global mean
np.random.seed(1000)
Y = X + np.random.randn(X.shape[0], X.shape[1])
t = np.arange(X.shape[0])

# get DMD modes and eigenvalues from first half of the data
svd_rank = 10
dmd = DMD(svd_rank=svd_rank)
n = X.shape[0]
dmd.fit(X.T)
# dmd.fit(X[:n//2].T)
# print(np.max(abs(dmd.reconstructed_data-X.T))) #18
# print(np.mean(abs(dmd.reconstructed_data-X.T))) #1
Theta = np.diag(dmd.eigs)
Psi = dmd.modes
nt = len(t)

def KF(k, Ps):
# define variance matrix of system disturbance and measurement noise
    Q = 10*np.eye(svd_rank)
    R = np.eye(k)

    # record
    u_errors = np.zeros(nt)
    post_covs = np.zeros((nt, svd_rank))

    # initial estimate and covariance matrix
    np.random.seed(123)
    a = np.linalg.pinv(Psi) @ X[0]
    sigma = abs(a)*0.3
    a += np.random.randn(svd_rank,2).view(np.complex128).flatten()*sigma/np.sqrt(2)
    np.random.seed()
    u = Psi @ a

    u_errors[0] = np.linalg.norm(u-X[0])/np.linalg.norm(X[0])
    Cov = np.eye(svd_rank) * sigma # init_cov
    post_covs[0] = np.diag(Cov)

    for i in range(1,nt):
        a = Theta @ a #predict, a priori
        Cov = Theta @ Cov @ Theta.conj().T + Q #cov, a priori

        P = Ps[(i-1)%len(Ps)]
        res = Y[i,P] - Psi[P] @ a
        # print(abs(res))
        S = Psi[P] @ Cov @ Psi[P].conj().T + R
        K = Cov @ Psi[P].conj().T @ np.linalg.pinv(S)
        a += K @ res # prediction, a posteriori
        Cov -= K @ Psi[P] @ Cov
        res = Y[i,P] - Psi[P] @ a
        # print(abs(res))

        u = Psi @ a
        u_errors[i] = np.linalg.norm(u-X[i])/np.linalg.norm(X[i])
        post_covs[i] = np.diag(Cov)

    return u_errors, np.sum(post_covs,axis=1)

#%% find moving sensor trajectories
k = 1
n_time = 14

grid = Grid2D(360,180,r=15, mask=mask, wrap_x=True)
model = OOQR(Psi, Theta, grid)
model.fit(n_time, k, cycle=True, restart=int(10/k))
Ps15 = model.P
u_errors1m15, post_covs1m15 = KF(k, Ps15)

plt.figure(figsize=(8,4))
plot_traj(Ps15)
plt.show()

plt.figure()
plt.plot(u_errors1m15, label='1 mobile (v=15)')
plt.show()

