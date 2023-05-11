import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import scipy.io
from scipy.linalg import qr
import pydmd
from grid import Grid1D
from OOQR import OOQR

# full dataset with dt=0.0001 on interval [0,10]
# spatial grid contains 2048 points on interval [0, 2*pi]
mat = scipy.io.loadmat('../../Datasets/ks/ks.mat')
# removed the first 200 transient data points
TT = mat['tt'][:,2000:]
UU = mat['uu'][:,2000:]
x = mat['x']
YY = UU + np.random.randn(UU.shape[0], UU.shape[1]) # add measurement noise

# plot
plt.figure(figsize=(4,4))
plt.imshow(UU[:,-10000:],extent=[9,10,0,2*np.pi],aspect=1/2/np.pi)
plt.xlabel('t',fontsize=13)
plt.ylabel('x',fontsize=13)
plt.tight_layout()
plt.show()

dmd=pydmd.DMD(svd_rank=100)
dmd.fit(UU)
# print(np.max(abs(dmd.reconstructed_data-UU))) # max disturbance: 72
# print(np.mean(abs(dmd.reconstructed_data-UU))) # mean disturbance: 16

# dmd modes and eigenvalues at dt=0.001
Psi = dmd.modes

# define Kalman filter process
def KF(Ps,dt_multiplier):
    k = Ps.shape[1]

    # define variance matrix of system disturbance and measurement noise
    Q = 1000*np.eye(Theta.shape[0]) #* dt_multiplier
    R = np.eye(k)

    # record
    u_errors = np.zeros(nt)
    post_covs = np.zeros(nt)

    # initial estimate and covariance matrix
    a = np.linalg.pinv(Psi) @ uu[:,0] + np.random.randn(100)
    u = Psi @ a
    U = np.zeros_like(uu)
    U[:,0] = np.real(u)
    u_errors[0] = np.linalg.norm(u-uu[:,0])/np.linalg.norm(uu[:,0])
    Cov = np.eye(100) * 100
    post_covs[0] = np.trace(Cov)

    for i in range(1,nt):
        a = Theta @ a #predict, a priori
        Cov = Theta @ Cov @ Theta.conj().T + Q #cov, a priori

        P = Ps[(i-1)%len(Ps)]
        res = yy[P,i] - Psi[P] @ a
        S = Psi[P] @ Cov @ Psi[P].conj().T + R
        K = Cov @ Psi[P].conj().T @ np.linalg.pinv(S)
        #print(a.shape, K.shape, res.shape)
        a += K @ res # prediction, a posteriori
        Cov -= K @ Psi[P] @ Cov
        res = yy[P,i] - Psi[P] @ a

        u = Psi @ a
        U[:,i] = np.real(u)
        u_errors[i] = np.linalg.norm(u-uu[:,i])/np.linalg.norm(uu[:,i])
        post_covs[i] = np.trace(Cov)

    sample = int(1e4//dt_multiplier)

    return U[:,-sample:], np.mean(u_errors[-sample:])
    # return post_covs

#%%
# change dt = 0.001
dt_multiplier = 10
tt = TT[:,::dt_multiplier]
uu = UU[:,::dt_multiplier]
yy = YY[:,::dt_multiplier]
Theta = np.diag(dmd.eigs**dt_multiplier)
nt = tt.size
k=10

grid = Grid1D(2048)
model = OOQR(Psi, Theta, grid)
model.fit(max(int(500/dt_multiplier),int(100/k)), k, r=10*dt_multiplier,
          cycle=True,restart=int(100/k))
Ps = model.P

Ui, error = KF(Ps,dt_multiplier)

plt.imshow(Ui[:,:100],extent=[9,9.1,0,2*np.pi],aspect=.1/2/np.pi)
for k in range(10):
    plt.plot(np.arange(9,9.1,0.001),Ps[np.arange(100)%50,k]/2048*2*np.pi,color='k')
plt.xlabel('t')
plt.ylabel('x')

