'''
Observability Optimizing QR (with grid_net)
---------------------------
Initialization:
    Psi - a basis matrix that maps high-dimensional data to low-dimensional representation
    Theta - a dynamics matrix in low-dimensional representation
    grid - a grid_net object that specifies all possible sensor locations and accessibility, including sensor constraints

'''

import numpy as np
from scipy.linalg import qr, svd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        self.n_loc, self.n_modes = Psi.shape
        self.grid = grid

    def _get_next_D(self, D):
        return D @ self.Theta

    def fit(self, n_time, n_sens, preset=None, cycle=False, restart=None):
        '''
        Fit sensor trajeoctories self.P (n_time x n_sens)
        Inputs:
            n_time - number of time steps in the tranjectory 
            n_sens - number of sensors
            preset - an array of predetermined sensor locations at some time steps
                     (can be used for fixed initial sensor location, refining trajectories at faster time scale in multiscale expansion, etc.)
            cycle - indicator whether the sensors return to the initial location at the end
            restart - number of time steps for restart strategy (useful when n_time > r)
        '''
        assert n_sens <= self.n_modes
        dest_time = n_time
        if preset is not None:
            dest_ind = np.where(np.any(preset!=-1, axis=1))[0]
            i_dest = 0
            dest_time=dest_ind[i_dest]
            self.grid.set_destination(preset[dest_ind[i_dest]], dest_ind[i_dest])

        if restart is None:
            restart=n_time

        n = restart * n_sens
        self.n_time = n_time
        self.n_sens = n_sens

        self.Obs = np.zeros((self.n_modes, n),dtype=complex)
        self.P = np.zeros((n_time, n_sens),dtype=int)
        ind=0

        constrained = (self.grid.r is not None)
        for i_time in range(n_time):
            if ind == 0:
                D = self.Psi.copy()
            else:
                D = self._get_next_D(D)

            if (preset is not None) and (len(preset) > i_time) and (preset[i_time] != -1).all():
                curr_p = preset[i_time]
                i_dest += 1
                if i_dest < len(dest_ind):
                    dest_time=dest_ind[i_dest]
                    self.grid.set_destination(preset[dest_ind[i_dest]], dest_time-i_time)
                else:
                    dest_time = n_time
                    self.grid.reset_destination()
            else:
                curr_p = np.zeros(self.n_sens,dtype=int)
                ns = None

                iter_start = 0
                if (i_time==0) or ((not constrained) and (ind*self.n_sens < self.n_modes)): # special case: underfit and no movement constraint, use qr directly
                    iter_start = min(self.n_sens, self.n_modes-ind*self.n_sens)
                    if ind == 0:
                        QD = D.T
                    else:
                        Q = qr(self.Obs[:,:ind*self.n_sens])[0]
                        QD = (Q.conj().T @ D.T)[ind*self.n_sens:,:]
                    p = qr(QD, pivoting=True)[2][:iter_start]
                    curr_p[:iter_start] = p
                    # print(p,curr_p)

                if constrained: # define variables needed for movement constraint
                    curr_origins = set()
                    prev = self.P[i_time-1]
                    prev_dict = {k: v for v, k in enumerate(prev)}
                    # print(prev_dict)
                    ns, ns_origin_dict = self.grid.get_neighbors(prev, dest_time-i_time)

                for j in range(iter_start, self.n_sens):
                    position = ind * self.n_sens + j

                    # find the scores for all candidates
                    if position < self.n_modes:
                        # qr with movement constraint
                        score = self._undersample(position, D.T, ns)

                    else:
                        # overfit, GappyPOD+E
                        score = self._oversample(position, D.T, ns)

                    # find the next location pj for sensor j
                    sort=np.argsort(-abs(score))
                    i=0
                    if ns is not None:
                        org = ns_origin_dict[ns[sort[i]]] - curr_origins
                        # print(score[sort[i]],ns[sort[i]],org)
                        # print(prev_dict)
                        # print(ns_origin_dict[ns[sort[i]]])
                        while len(org) == 0:
                            i += 1
                            org = ns_origin_dict[ns[sort[i]]] - curr_origins
                            # print(score[sort[i]],ns[sort[i]],org)
                        pj = ns[sort[i]]
                        ns.remove(pj)

                        if len(org) == 1:   # if only one origin, then matched automatically
                            curr_origins.update(org)
                            org_idx = prev_dict[list(org)[0]]
                        else:               # if more than one origins, randomly assign to one
                            temp = np.random.choice(list(org))
                            curr_origins.update({temp})
                            org_idx = prev_dict[temp]
                        curr_p[org_idx] = pj

                    else:
                        while sort[i] in curr_p[:j]:
                            i += 1
                        pj = sort[i]
                        curr_p[j] = pj

                    self.Obs[:,position] = D[pj]

            # print(i_time, curr_p,preset[dest_ind[i_dest]],dest_time)
            self.P[i_time] = curr_p
            self.Obs[:, ind * self.n_sens:(ind+1) * self.n_sens] = D[curr_p].T

            if cycle and not self.grid.dest:
                if i_time == 0:
                    self.grid.set_destination(curr_p, int(self.n_time//2))
                else:
                    dest_time=self.n_time
                    self.grid.set_destination(self.P[0], self.n_time-i_time)

            ind += 1
            if ind >= restart and ind < n_time-1: #restart
                self.Obs = np.zeros((self.n_modes, n),dtype=complex)
                D = self.Psi.copy()
                self.Obs[:,:self.n_sens] = D[curr_p].T
                ind=1

    def _undersample(self, obs_ind, A, ns=None):
        '''
        Compute the scores for one step of underfitting selection (QRcp).

        Inputs: obs_ind - current index in observability matrix
                A - matrix to sample from
                ns - list of candidate indices

        '''
        Q = qr(self.Obs[:,:obs_ind])[0]
        Ab = (Q.conj().T @ A)[obs_ind:,:]
        if ns is not None:
            Ab = Ab[:,ns]
        score = np.linalg.norm(Ab, axis=0)**2
        return score

    def _oversample(self, obs_ind, A, ns=None):
        '''
        Compute the scores for one step of overfitting selection (B. Peherstorfer, Z. Drmac, S. Gugercin, 2020).

        Inputs: obs_ind - current index in observability matrix
                A - matrix to sample from
                ns - list of candidate indices

        '''
        _,S,W = svd(self.Obs[:,:obs_ind].T, full_matrices=False)
        g = S[-2]**2 - S[-1]**2
        Ab = W.conj().T @ A.conj()
        if ns is not None:
            Ab = Ab[:,ns]
        score = g + np.linalg.norm(Ab, axis=0)**2
        score -= np.sqrt(score ** 2 - 4 * g * abs(Ab[-1])**2)
        return score

    def fit_random(self, n_time, n_sens, cycle=False, restart=None):
        if restart is None:
            restart=n_time
        assert n_sens <= self.n_modes
        n = restart * n_sens
        self.n_time = n_time
        self.n_sens = n_sens

        self.Obs = np.zeros((self.n_modes, n),dtype=complex)
        self.P = np.zeros((n_time, n_sens),dtype=int)
        self.P[0] = np.random.choice(self.n_loc, n_sens, replace=False)

        constrained = (self.grid.r is not None)
        if cycle:
            self.grid.set_destination(self.P[0], int(self.n_time//2))

        for i_time in range(1,n_time):
            curr_p = np.zeros(self.n_sens,dtype=int)

            if constrained: # define variables needed for movement constraint
                curr_origins = set()
                prev = self.P[i_time-1]
                prev_dict = {k: v for v, k in enumerate(prev)}
                # print(prev_dict)
                ns, ns_origin_dict = self.grid.get_neighbors(prev, self.n_time-i_time)

                sort = np.random.permutation(ns)
                j = 0
                for p in sort:
                    org = ns_origin_dict[p] - curr_origins
                    if len(org) == 0:
                        continue
                    elif len(org) == 1:
                        curr_origins.update(org)
                        org_idx = prev_dict[list(org)[0]]
                    else:
                        temp = np.random.choice(list(org))
                        curr_origins.update({temp})
                        org_idx = prev_dict[temp]
                    curr_p[org_idx] = p
                    j += 1
                    if j == self.n_sens:
                        break

            else:
                curr_p = np.random.choice(self.n_loc, n_sens, replace=False)

            self.P[i_time] = curr_p


    # depreciated
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
        prev=None
        while True:
            s = min(tf-i, n_time)
            if random:
                self.fit_random(s, n_sens, r=r, preset=prev)
            else:
                self.fit(s, n_sens, r=r, preset=prev)

            self.trajectory[i:i+s] = self.P

            if i + s >= tf:
                break

            prev = self.P[dt:]
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


