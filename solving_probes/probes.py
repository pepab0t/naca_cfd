import re

import matplotlib.pyplot as plt
import numpy as np


def plot_field(xyz: np.ndarray, show:bool=True) -> None:
    shp = int(np.sqrt(len(xyz)))
    
    X = xyz[:,0].reshape(shp, shp)
    Y = xyz[:,1].reshape(shp, shp)
    Z = xyz[:,2].reshape(shp, shp)

    plt.contourf(X, Y, Z, 100, cmap='coolwarm')
    plt.colorbar()
    plt.axis('equal')
    plt.show()

def read_probes():
    sel_line = ''
    with open('./p', 'r') as f:
        for line in f:
            if line.strip().startswith('500'):
                sel_line = line.strip()
                break
    
    sel_line = sel_line.replace('-1e+300', '10000')
    vals = re.findall(r'(\S+)',sel_line)
    vals.remove('500')

    return np.array([float(x) for x in vals])

def main():
    a: np.ndarray = np.zeros((4096,3))

    i = 0
    with open('probes') as f:
        for line in f:
            if re.search(r'.*\d+\s\-?\d+\s\-?\d+', line):
                match = re.findall(r'[\-0-9\.]+', line)
                a[i, 0], a[i, 1] = float(match[0]), float(match[2])
                i += 1

    a[:, 2] = read_probes()

    print(len(a))
    plot_field(a)

    

if __name__ == '__main__':
    main()
