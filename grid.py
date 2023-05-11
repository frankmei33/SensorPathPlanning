'''
Create a 1D/2D grid object.
'Mask' marks any obstacles on the grid. 
It is an array of the same size as the grid, where a location is True if clear and False is blocked.
'Wrap' marks whether the grid is wrapped in a direction so the sensor can move directly from one end to the other.
'''

import numpy as np
from scipy.linalg import qr, svd

class Grid1D():
    def __init__(self, length, mask=None, wrap=False):
        if mask is not None:
            assert length == mask.size
        self.length = length
        self.mask = mask
        self.wrap = wrap
        self.size = length if mask is None else np.sum(mask)

    def ind_to_loc(self, ind):
        if self.mask is not None:
            x = np.where(self.mask==True)
            return x[ind]
        else:
            return ind

    def loc_to_ind(self, loc):
        if self.mask is not None:
            return np.sum(self.mask[:loc])
        else:
            return loc

    def get_neighbors_single(self, x, r, destination=None, steps_left=None):
        if self.wrap:
            if r >= self.length/2:
                x_min, x_max = 0, self.length-1
            else:
                x_min, x_max = x-r, x+r
        else:
            x_min, x_max = max(0,x-r), min(self.length-1,x+r)

        neighbors=[]
        assert x_max-x_min < self.length
        for i in range(x_min, x_max+1):
            if (destination is not None) and (abs(i-destination)>r*steps_left):
                continue

            if (self.mask is not None) and not self.mask[i]:
                continue

            neighbors.append(i % self.length)

        return neighbors

    def get_neighbors(self, inds, r, destination=None, steps_left=None):
        if destination is not None:
            assert len(inds) == len(destination)

        neighbors = set()
        neighbor_dict = dict()
        for i in range(len(inds)):
            if destination is not None:
                ns_ind = self.get_neighbors_single(inds[i], r, destination[i], steps_left)
            else:
                ns_ind = self.get_neighbors_single(inds[i], r)
            neighbors.update(ns_ind)
            for n in ns_ind:
                if n in neighbor_dict:
                    neighbor_dict[n].update({inds[i]})
                else:
                    neighbor_dict[n] = {inds[i]}

        return list(neighbors), neighbor_dict

class Grid2D():
    '''
    Indexing the grid from the top left corner.
    For example, a grid with width 3 and height 2 is
        1,2,3,
        4,5,6.

    '''
    def __init__(self, width, height, mask=None, wrap_x=False, wrap_y=False):
        if mask is not None:
            assert width * height == mask.size
        self.width = width # x direction
        self.height = height # y direction
        self.mask = mask
        self.wrap_x = wrap_x
        self.wrap_y = wrap_y
        self.size = width * height if mask is None else np.sum(mask)

    def ind_to_loc(self, ind):
        if self.mask is not None:
            y, x = np.where(self.mask==True)
            return (x[ind], y[ind])
        else:
            return (ind % self.width, ind // self.width)

    def loc_to_ind(self, loc):
        val = (loc[0] % self.width) + (loc[1] % self.height) * self.width
        if self.mask is not None:
            return np.sum(self.mask.flatten()[:val])
        else:
            return val

    def dist(self, loc1, loc2):
        dx = abs(loc1[0]-loc2[0])
        if self.wrap_x:
            dx = min(dx, self.width-dx)
        dy = abs(loc1[1]-loc2[1])
        if self.wrap_y:
            dy = min(dy, self.height-dy)
        return np.sqrt(dx**2+dy**2)

    # check if there is obstacles on the straight path from loc1 to loc2
    def has_obs(self, loc1, loc2):
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

    def get_neighbors_single(self, ind, r, destination=None, steps_left=None):
        origin = self.ind_to_loc(ind)
        x,y = origin
        if destination is not None:
            dest_loc = self.ind_to_loc(destination)
        x_min = x-r if self.wrap_x else max(0,x-r)
        x_max = x+r if self.wrap_x else min(self.width-1,x+r)
        y_min = y-r if self.wrap_y else max(0,y-r)
        y_max = y+r if self.wrap_y else min(self.height-1,y+r)

        neighbors = list()
        for i in range(x_min,x_max+1):
            for j in range(y_min,y_max+1):
                loc = (i%self.width, j%self.height)

                # within radius r
                if self.dist(loc, origin) > r:
                    continue

                # able to reach destination within steps left:
                if (destination is not None) and (self.dist(loc, dest_loc) > (r*steps_left)):
                    continue

                # mask/obstacle
                if (self.mask is not None) and not self.mask[loc[1],loc[0]]:
                    continue

                if (self.mask is not None) and self.has_obs(origin, loc):
                    continue

                neighbors.append(loc)

        # neighbors = [(i % self.width,j % self.height) for i in range(x_min,x_max+1)
        #                 for j in range(y_min,y_max+1) if (x-i)**2 + (y-j)**2 <= r**2]
        # if self.mask is not None:
        #     neighbors = [loc for loc in neighbors if self.mask[loc[1],loc[0]]]

        #     # add method here to detect mask/obstacle

        # if start is not None:
        #     neighbors = [loc for loc in neighbors if (loc[0]-start_x)**2 + (loc[1]-start_y)**2 < (r*steps_left)**2]
        ns_ind = [self.loc_to_ind(n) for n in neighbors]

        return ns_ind

    def get_neighbors(self, inds, r, destination=None, steps_left=None):
        if destination is not None:
            assert len(inds) == len(destination)

        neighbors = set()
        neighbor_dict = dict()
        for i in range(len(inds)):
            if destination is not None:
                ns_ind = self.get_neighbors_single(inds[i], r, destination[i], steps_left)
            else:
                ns_ind = self.get_neighbors_single(inds[i], r)
            neighbors.update(ns_ind)
            for n in ns_ind:
                if n in neighbor_dict:
                    neighbor_dict[n].update({inds[i]})
                else:
                    neighbor_dict[n] = {inds[i]}

        return list(neighbors), neighbor_dict

