'''
Create a 1D/2D grid object as a connected network.
'Mask' marks any obstacles on the grid. 
It is an array of the same size as the grid, where a location is True if clear and False is blocked.
'Wrap' marks whether the grid is wrapped in a direction so the sensor can move directly from one end to the other.
Internally always uses index (ind) instead of coordinate (loc).
'''

import numpy as np
from scipy.linalg import qr, svd
from collections import deque

class Grid():
    dest = False

    def bfs(self, src_ind, max_dist):
        '''Use Breadth-first search to get the shortest path from x to all nodes within a max distance.'''
        queue = deque()
        visited = [False for ind in range(self.size)]
        dist = [10**9 for ind in range(self.size)]
        visited[src_ind] = True
        dist[src_ind] = 0
        queue.append(src_ind)

        while len(queue) != 0:
            ind = queue.popleft()
            if dist[ind] >= max_dist:
                continue

            for i in range(len(self.adj[ind])):
                temp = self.adj[ind][i] # adjacent node index
                if not visited[temp]:
                    visited[temp] = True
                    dist[temp] = dist[ind] + 1
                    queue.append(temp)

        return dist

    def set_destination(self, inds, max_dist=None):
        '''Set the ending sensor positions, construct shortest path lists for sensors.
           Max distance is the time step to reach the ending position.
           If the ending position is the same at where it starts, it is a cycle.
           Max distance is half of the cycle period.
        '''
        # self.start = inds
        self.dist_to_dest = np.zeros((len(inds), self.size))
        self.dest = True

        for i in range(len(inds)):
            self.dist_to_dest[i] = self.bfs(inds[i], max_dist)

    def reset_destination(self):
        self.dist_to_dest = None
        self.dest = False

    def set_radius(self, r):
        self.adj = dict()
        for ind in range(self.size):
            self.adj[ind] = self.get_adj(ind)

    # def set_start(self, inds, max_dist=None):
    #     '''Set the ending sensor positions, construct shortest path lists for sensors.
    #        Max distance is the time step to reach the ending position.
    #        If the ending position is the same at where it starts, it is a cycle.
    #        Max distance is half of the cycle period.
    #     '''
    #     # self.start = inds
    #     self.dist_from_start = np.zeros((len(inds), self.size))
    #     self.cycle = True

    #     for i in range(len(inds)):
    #         self.dist_from_start[i] = self.bfs(inds[i], max_dist)

    def get_neighbors(self, inds, nbg_dist_to_start=None):
        if nbg_dist_to_start is None:
            nbg_dist_to_start = 1e9
        neighbors = set()
        neighbor_dict = dict()
        for i in range(len(inds)):
            if self.dest:
                ns_ind = [j for j in self.adj[inds[i]] if self.dist_to_dest[i, j] <= nbg_dist_to_start]
            else:
                ns_ind = self.adj[inds[i]]

            if len(ns_ind) == 0:
                raise ValueError("No valid neighbor is found for sensor {}.".format(i+1))

            neighbors.update(ns_ind)
            for n in ns_ind:
                if n in neighbor_dict:
                    neighbor_dict[n].update({inds[i]})
                else:
                    neighbor_dict[n] = {inds[i]}

        return list(neighbors), neighbor_dict

    def get_adj(self, ind):
        pass


class Grid1D(Grid):
    def __init__(self, length, r=None, mask=None, wrap=False):
        '''Initialize. Build an adjacency dictionary. '''
        if mask is not None:
            assert length == mask.size
        self.length = length
        self.mask = mask
        self.wrap = wrap
        self.size = length if mask is None else np.sum(mask)
        self.r = r
        if r is not None:
            self.set_radius(r)

    def ind_to_loc(self, ind):
        if self.mask is not None:
            x = np.where(self.mask==True)[0]
            return x[ind]
        else:
            return ind

    def loc_to_ind(self, loc):
        if self.mask is not None:
            return np.sum(self.mask[:loc])
        else:
            return loc

    def get_adj(self, ind):
        x = self.ind_to_loc(ind)
        x_min = x-self.r if self.wrap else max(0,x-self.r)
        x_max = x+self.r if self.wrap else min(self.length-1,x+self.r)

        neighbors = [(i % self.length) for i in range(x_min,x_max+1)]
        if self.mask is not None:
            neighbors = [self.loc_to_ind(loc) for loc in neighbors if self.mask[loc]]
        return neighbors

class Grid2D(Grid):
    '''
    Indexing the grid from the top left corner.
    For example, a grid with width 3 and height 2 is
        1,2,3,
        4,5,6.

    '''
    def __init__(self, width, height, r=None, mask=None,
                 wrap_x=False, wrap_y=False, dist_type='euclidean'):
        if mask is not None:
            assert width * height == mask.size
        self.width = width # x
        self.height = height # y
        self.mask = mask
        self.wrap_x = wrap_x
        self.wrap_y = wrap_y
        self.dist_type = dist_type
        self.size = width * height if mask is None else int(np.sum(mask))
        self.r = r
        if r is not None:
            self.set_radius(r)
        else:
            self.adj = dict()
            for ind in range(self.size):
                self.adj[ind] = list(range(self.size))


    def ind_to_loc(self, ind):
        if self.mask is not None:
            y, x = np.where(self.mask==True)
            return (x[ind], y[ind])
        else:
            return (ind % self.width, ind // self.width)

    def loc_to_ind(self, loc):
        x = loc[0] % self.width
        y = loc[1] % self.height
        val = x + y * self.width
        if self.mask is not None:
            if self.mask[y,x]:
                return int(np.sum(self.mask.flatten()[:val]))
            else:
                return None
        else:
            return val

    def dist(self, loc1, loc2):
        if self.dist_type=='euclidean':
            dx = abs(loc1[0]-loc2[0])
            if self.wrap_x:
                dx = min(dx, self.width-dx)
            dy = abs(loc1[1]-loc2[1])
            if self.wrap_y:
                dy = min(dy, self.height-dy)
            return np.sqrt(dx**2+dy**2)
        elif self.dist_type=='sphere':
            lat1 = (90-loc1[1]) * np.pi / 180
            lat2 = (90-loc2[1]) * np.pi / 180
            dx = abs(loc1[0]-loc2[0])
            dx = min(dx, self.width-dx)
            dlong = dx * np.pi / 180
            R = 6371
            d = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlong)) * R
            return d

        return None

    def has_obs(self, loc1, loc2):
        if self.mask is None:
            return False

        x_range = np.sort([loc1[0], loc2[0]])
        y_range = np.sort([loc1[1], loc2[1]])
        dist_x = x_range[1]-x_range[0]
        dist_y = y_range[1]-y_range[0]
        if self.wrap_x and dist_x > (self.width-dist_x):
            x_range = np.array([x_range[1],x_range[0]+self.width])
            dist_x = x_range[1]-x_range[0]
        if self.wrap_y and dist_y > (self.height-dist_y):
            y_range = np.array([y_range[1],y_range[0]+self.height])
            dist_y = y_range[1]-y_range[0]
        if dist_x > dist_y:
            for i in range(dist_x):
                x = int(x_range[0] + i + 1)
                y = int(y_range[0] + np.round((i+1)/dist_x*dist_y))
                if not self.mask[y%self.height, x%self.width]:
                    return True
        else:
            for i in range(dist_y):
                y = int(y_range[0] + i + 1)
                x = int(x_range[0] + np.round((i+1)/dist_y*dist_x))
                if not self.mask[y%self.height, x%self.width]:
                    return True
        return False

    def get_adj(self, ind):
        origin = self.ind_to_loc(ind)
        x,y = origin
        x_min = x-self.r if self.wrap_x else max(0,x-self.r)
        x_max = x+self.r if self.wrap_x else min(self.width-1,x+self.r)
        y_min = y-self.r if self.wrap_y else max(0,y-self.r)
        y_max = y+self.r if self.wrap_y else min(self.height-1,y+self.r)

        neighbors = list()
        for i in range(x_min,x_max+1):
            for j in range(y_min,y_max+1):
                loc = (i%self.width, j%self.height)

                # within radius r
                if self.dist(loc, origin) > self.r:
                    continue

                # mask/obstacle
                if (self.mask is not None) and (not self.mask[loc[1],loc[0]]):
                    continue

                if (self.mask is not None) and (self.has_obs(origin, loc)):
                    continue

                neighbors.append(loc)

        ns_ind = [self.loc_to_ind(n) for n in neighbors]

        return ns_ind
