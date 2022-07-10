import numpy as np
import matplotlib.pyplot as plt
import re

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
            
def main():
    P = PointMaker(top=1, bottom=-1, left=-1, right=1, n_h=8, n_v=8)
    points = P.make()
    P.display(points)
    P.to_probes(points, test_mode=True)

if __name__ == "__main__":
    main()