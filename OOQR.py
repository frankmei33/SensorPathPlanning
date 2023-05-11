'''
Observability Optimizing QR
---------------------------
Inputs:
    Psi - a basis matrix that maps high-dimensional data to low-dimensional representation
    Theta - a dynamics matrix in low-dimensional representation
    grid - a grid object that specifies all possible sensor locations and accessibility, including sensor constraints

'''

import numpy as np
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from grid import *

class OOQR():
    def __init__(self, Psi, Theta, grid):
        '''
        Psi : basis, n-by-r
        Theta : low rank dynamics, r-by-r
        '''
        assert Psi.shape[0] == grid.size
        assert Psi.shape[1] == Theta.shape[0]

        self.Psi = Psi
        self.Theta = Theta
        #self.t = t
        self.n_loc, self.n_modes = Psi.shape
        self.grid = grid

    def _get_next_D(self, D):
        return D @ self.Theta

    # fit sensor trajeoctories self.P (n_time x n_sens)
    def fit(self, n_time, n_sens, r=None, preset=None, cycle=False, restart=None):
        if restart is None:
            restart = n_time
        assert n_sens <= self.n_modes
        n = restart * n_sens
        self.n_time = n_time
        self.n_sens = n_sens

        self.Obs = np.zeros((self.n_modes, n),dtype=complex)
        self.P = np.zeros((n_time, n_sens),dtype=int)
        ind=0
        if preset is not None:
            dest_ind = np.where(np.any(preset!=-1, axis=1))[0]
            i_dest = 0
        dest = None

        for i_time in range(n_time):
            # tt = self.t[i_time]
            if ind == 0:
                D = self.Psi
            else:
                D = self._get_next_D(D) # current data matrix

            if preset is not None:
                if i_dest >= len(dest_ind):
                    dest = None
                else:
                    dest = (preset[dest_ind[i_dest]], dest_ind[i_dest])

            if preset is not None and len(preset) > i_time and (preset[i_time] != -1).all():
                p = preset[i_time]
                i_dest += 1
            else:
                if ind == 0:
                    p = qr(D.T, pivoting=True)[2][:self.n_sens]
                elif ind * self.n_sens < self.n_modes: # undersample, use QR
                    p = self._choose_qr(D.T, ind, i_time, r, cycle, dest)
                else: # oversample
                    p = self._choose_oversample(D.T, ind, i_time, r, cycle, dest)

            self.P[i_time] = p
            self.Obs[:, ind * self.n_sens:(ind+1) * self.n_sens] = D[p].T

            ind += 1
            if ind >= restart: #restart
                self.Obs = np.zeros((self.n_modes, n),dtype=complex)
                D = self.Psi.copy()
                self.Obs[:,:self.n_sens] = D[p].T
                ind=1

    # fit random paths that satisfy constraints
    def fit_random(self, n_time, n_sens, r=None, preset=None, cycle=False):
        n = n_time * n_sens
        self.n_time = n_time
        self.n_sens = n_sens

        self.Obs = np.zeros((self.n_modes, n),dtype=complex)
        self.P = np.zeros((n_time, n_sens),dtype=int)

        for i_time in range(n_time):
            # tt = self.t[i_time]
            if i_time == 0:
                D = self.Psi
            else:
                D = self._get_next_D(D) # current data matrix

            if preset is not None and len(preset) > i_time:
                p = preset[i_time]
            else:
                if i_time == 0:
                    p = np.random.choice(self.n_loc, self.n_sens)
                else:
                    p = self._choose_random(i_time, r, cycle)

            self.P[i_time] = p
            self.Obs[:, i_time * self.n_sens:(i_time+1) * self.n_sens] = D[p].T
    
    # underfitting selection with QR
    def _choose_qr(self, A, ind, i_time, r, cycle=False, dest=None):
        # dest is a tuple (dest location, time step of dest)
        assert ind > 0

        Q = qr(self.Obs[:,:ind*self.n_sens])[0]
        A = (Q.conj().T @ A)[ind*self.n_sens:,:]

        if r is None:
            return qr(A, pivoting=True)[2][:self.n_sens]

        m,n = A.shape
        preset = self.P[i_time-1]
        preset_dict = {k: v for v, k in enumerate(preset)}
        if dest is not None:
            ns, ns_origin_dict = self.grid.get_neighbors(preset, r, dest[0], dest[1]-i_time)
        elif cycle:
            ns, ns_origin_dict = self.grid.get_neighbors(preset, r, self.P[0], self.n_time-i_time)
        else:
            ns, ns_origin_dict = self.grid.get_neighbors(preset, r)
        if len(ns) == 0:
            print(preset, r, dest, i_time,self.n_time-i_time,self.P[0])
            raise ValueError('No neighbor candidates that match constraints are found.')

        curr_origins = set()
        # temp_dict = dict()
        colnorms = -np.ones(A.shape[1]) * np.inf
        colnorms[ns] = np.linalg.norm(A[:,ns], axis=0)**2
        curr = np.zeros(self.n_sens,dtype=int)

        # perm = np.arange(n)
        # tau = np.zeros(n_sens,dtype=A.dtype)
        # F = np.zeros((n, n_sens),dtype=A.dtype)
        for j in range(self.n_sens):
            sort=np.argsort(-colnorms)
            i=0
            org = ns_origin_dict[sort[i]] - curr_origins
            while len(org) == 0:
                # print('overlap:', perm[sort[i]])
                colnorms[sort[i]] = -np.inf
                i += 1
                org = ns_origin_dict[sort[i]] - curr_origins
            p = sort[i]

            # if only one origin, then matched automatically
            if len(org) == 1:
                curr_origins.update(org)
                org_idx = preset_dict[list(org)[0]]
                curr[org_idx] = p
            # if more than one origins
            else:
                # randomly assign to one
                temp = np.random.choice(list(org))
                curr_origins.update({temp})
                org_idx = preset_dict[temp]
                curr[org_idx] = p

            Qnew = qr(A[j:,[p]])[0]
            A[j:] = Qnew.conj().T @ A[j:]
            colnorms -= np.abs(A[j])**2

        # print(curr)
        return curr

    # overfitting selection
    # (B. Peherstorfer, Z. Drmac, S. Gugercin, 2020)
    def _choose_oversample(self, A, ind, i_time, r, cycle=False, dest=None):
        m,n = A.shape
        preset = self.P[i_time-1]
        preset_dict = {k: v for v, k in enumerate(preset)}
        if r is not None:
            if dest is not None:
                ns, ns_origin_dict = self.grid.get_neighbors(preset, r, dest[0], dest[1]-i_time)
            elif cycle:
                ns, ns_origin_dict = self.grid.get_neighbors(preset, r, self.P[0], self.n_time-i_time)
            else:
                ns, ns_origin_dict = self.grid.get_neighbors(preset, r)
            if len(ns) == 0:
                raise ValueError('No neighbor candidates that match constraints are found.')

        curr_origins = set()
        curr = np.zeros(self.n_sens,dtype=int)
        preset_X = self.Obs[:,:(ind+1) * self.n_sens]

        for j in range(self.n_sens):
            _,S,W = svd(preset_X[:,:ind * self.n_sens + j].T, full_matrices=False)
            g = S[-2]**2 - S[-1]**2
            Ab = W.conj().T @ A.conj()
            if r is not None:
                # score = np.zeros(A.shape[1],dtype=complex)
                score = g + np.linalg.norm(Ab[:,ns], axis=0)**2
                score -= np.sqrt(score ** 2 - 4 * g * abs(Ab[-1, ns])**2)
                sort=np.argsort(-abs(score))
                i=0

                org = ns_origin_dict[ns[sort[i]]] - curr_origins
                while len(org) == 0:
                    i += 1
                    org = ns_origin_dict[ns[sort[i]]] - curr_origins
                p = ns[sort[i]]
                ns.remove(p)
            else:
                score = g + np.linalg.norm(Ab, axis=0)**2
                score -= np.sqrt(score ** 2 - 4 * g * abs(Ab[-1])**2)
                sort = np.argsort(-abs(score))
                i = 0
                while sort[i] in curr:
                    i += 1
                p = sort[i]
                org = set(preset) - curr_origins

            preset_X[:, ind * self.n_sens + j] = A[:,p]

            # if only one origin, then matched automatically
            if len(org) == 1:
                curr_origins.update(org)
                org_idx = preset_dict[list(org)[0]]
                curr[org_idx] = p
            # if more than one origins
            else:
                # randomly assign to one
                temp = np.random.choice(list(org))
                curr_origins.update({temp})
                org_idx = preset_dict[temp]
                curr[org_idx] = p

        return curr

    # random selection
    def _choose_random(self, ind, r, cycle=False):
        preset = self.P[ind-1]

        if r is None:
            return np.random.choice(self.n_loc, len(preset))

        curr = np.zeros(self.n_sens,dtype=int)
        for i in range(self.n_sens):
            if cycle:
                ns = self.grid.get_neighbors_single(preset[i], r, self.P[0,i], self.n_time-ind)
            else:
                ns = self.grid.get_neighbors_single(preset[i], r)

            temp = np.random.choice(ns)
            while temp in curr[:i]:
                temp = np.random.choice(ns)
            curr[i] = temp
        return curr


    def estimate_b(self, data, P = None):
        # if no P is provided, then use the model fit P
        if P is None:
            n = self.Obs.shape[1]
            y = np.zeros(n,dtype=complex)
            for i_time in range(self.n_time):
                i_sens = i_time*self.n_sens
                p = self.P[i_time]
                y[i_sens: i_sens+self.n_sens] = data[i_time,p]
            b_est = y @ np.linalg.pinv(self.Obs)
        else:
            n = len(P.flatten())
            y = np.zeros(n,dtype=complex)
            X = np.zeros((self.n_modes, n),dtype=complex)
            for i_time in range(self.n_time):
                i_sens = i_time*self.n_sens
                p = P[i_time]
                y[i_sens: i_sens+self.n_sens] = data[i_time,p]
                if i_time == 0:
                    AA = self.Psi
                else:
                    AA = self._get_next_D(AA)
                X[:, i_time * self.n_sens:(i_time+1) * self.n_sens] = AA[:,p]
            b_est = y @ np.linalg.pinv(X)

        return b_est

    def reconstruct(self, data):
        b_est = self.estimate_b(data)
        recon = np.zeros_like(data)
        for i in range(len(data)):
            recon[i] = self.Psi @ b_est
            b_est = self.Theta @ b_est
        return recon


    def fit_traj(self, tf, dt, n_time, n_sens, r=None, random=False):
        assert dt < n_time
        self.trajectory = np.zeros((tf, n_sens),dtype=int)

        i = 0
        preset=None
        while True:
            s = min(tf-i, n_time)
            if random:
                self.fit_random(s, n_sens, r=r, preset=preset)
            else:
                self.fit(s, n_sens, r=r, preset=preset)

            self.trajectory[i:i+s] = self.P

            if i + s >= tf:
                break

            preset = self.P[dt:]
            i += dt

    # depreciated, to be updated
    def animate_traj(self, canvas):
        tf = self.trajectory.shape[0]
        # Plotting
        fig = plt.figure(figsize=(8,5))
        ax = fig.subplots()

        snapshot = float("nan")*np.ones((self.grid.height,self.grid.width))
        snapshot[self.grid.mask] = canvas
        ax.imshow(snapshot, cmap='coolwarm')

        locs = np.array([list(self.grid.ind_to_loc(i)) for i in self.trajectory[0]])
        p = ax.scatter(locs[:,0], locs[:,1], c='yellow', edgecolors='black')
        ax.set_title('t=0')

        # Updating the plot
        def update(val):
            locs = np.array([list(self.grid.ind_to_loc(i)) for i in self.trajectory[val]])
            p.set_offsets(locs)
            ax.set_title('t={}'.format(val))
            #redrawing the figure
            fig.canvas.draw()

        ani = FuncAnimation(fig, update, frames=np.arange(tf),interval=200)

        return ani

