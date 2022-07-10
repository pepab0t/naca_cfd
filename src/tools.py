import numpy as np
import matplotlib.pyplot as plt
import re
import concurrent.futures
from src.entities import Line, Vector
from src.NacaGenerator import NacaProfile


class PointMaker:
    probes_filepath = '../NacaAirfoil/system/probes'

    def __init__(self, top, bottom, left, right, n_v=8, n_h=8):
        self.start_h = left
        self.start_v = bottom
        self.n_h = n_h
        self.n_v = n_v
        self.dx = (right - left) / (n_h-1) 
        self.dy = (top - bottom) / (n_v-1)
        # self.points = np.zeros(shape=(n_v, n_h, 2))

        # self.make()

    def make(self) -> np.array:
        points = np.zeros(shape=(self.n_v, self.n_h, 2))
        for i in range(self.n_v):
            for j in range(self.n_h):
                x = self.start_h + j * self.dx
                z = self.start_v + i * self.dy
                points[i, j, 0] = x
                points[i, j, 1] = z

        return points

    def display(self, points: np.array, show:bool = False) -> None:
        plt.plot(points[:,:,0].reshape(-1), points[:,:,1].reshape(-1), 'o')
        if show:
            plt.show()

    def to_probes(self, points: np.array,test_mode: bool=False, repr: bool=False) -> None:
        out = ''
        for line in open(self.probes_filepath, 'r'):
            out += line
            if 'probeLocations' in line:
                break
        out += '(\n'

        for row in points:
            for p in row:
                # print(p)
                out += '\t' + f"({p[0]} 0 {p[1]})" + '\n'

        out += ');'

        if repr:
            print(out)

        if not test_mode:
            with open(self.probes_filepath, 'w') as f:
                f.write(out)

class DistField:

    def __init__(self, profile: NacaProfile, point_field: np.ndarray):
        """point_field needs to be in shape (n1,n2,2)"""

        self.naca = profile
        self.point_field = point_field.reshape(point_field.shape[0] * point_field.shape[1], point_field.shape[2])

    def evaluate(self) -> np.ndarray:

        distances = np.zeros(len(self.point_field))
        for i in range(len(self.point_field)):
            distances[i] = self.shortest_distance(self.point_field[i,:])

        out: np.ndarray = np.zeros(shape=(self.point_field.shape[0], self.point_field.shape[1]+1))
        out[:,0:2] = self.point_field.copy()
        out[:,2] = distances

        return out

    def evaluate_parallel(self) -> np.ndarray:

        with concurrent.futures.ProcessPoolExecutor() as executor:
            procs = [executor.submit(self.shortest_distance, point) for point in self.point_field]

        out: np.ndarray = np.zeros(shape=(self.point_field.shape[0], self.point_field.shape[1]+1))
        out[:,0:2] = self.point_field.copy()
        
        for i in range(len(procs)):
            out[i,2] = procs[i].result()
            
        return out

    # def evaluate_parallel_cy(self):

    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         procs = [executor.submit(shortest_distance_cy, self.naca.upper, self.naca.lower, point) for point in self.point_field]

    #     out = np.zeros(shape=(self.point_field.shape[0], self.point_field.shape[1]+1))
    #     out[:,0:2] = self.point_field.copy()

    #     for i in range(len(procs)):
    #         out[i,2] = procs[i].result()

    #     return out

    def shortest_distance(self, point: np.ndarray) -> float:
        Line._validate_entity(point)

        d = 1e6

        for i in range(1, len(self.naca.upper)):
            l = Line(self.naca.upper[i-1,:], self.naca.upper[i,:])
            d_new = l.distance_segment(point)
            if d_new < d:
                d = d_new 

        for i in range(1, len(self.naca.lower)):
            l = Line(self.naca.lower[i-1,:], self.naca.lower[i,:])
            d_new = l.distance_segment(point)
            if d_new < d:
                d = d_new

        d = (-1)**int(not self.naca.is_outside(point)) * d

        return d

    def plot_input(self, xyz: np.ndarray, show:bool=True) -> None:
        shp = int(np.sqrt(len(xyz)))

        profile: np.ndarray = np.concatenate([self.naca.upper[::-1], self.naca.lower])

        X = xyz[:,0].reshape(shp, shp)
        Y = xyz[:,1].reshape(shp, shp)
        Z = xyz[:,2].reshape(shp, shp)

        plt.contourf(X, Y, Z, 100, cmap='coolwarm')
        plt.colorbar()

        plt.plot(profile[:,0], profile[:,1], '-k')

        if show:
            plt.show()

def main():
    P = PointMaker(top=1, bottom=-1, left=-1, right=1, n_h=8, n_v=8)
    points = P.make()
    P.display(points)
    P.to_probes(points, test_mode=True)

if __name__ == "__main__":
    main()