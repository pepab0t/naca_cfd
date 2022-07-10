import numpy as np
from MakePoints import PointMaker
from entities import Vector, Line
from NacaGenerator import NacaProfile
import time

def test1():
    v1 = np.array([2,0])

    v = np.array([1,1])

    phi = 45 * np.pi / 180
    l = 1

    u = np.ones(2)

    u[0] = np.cos(phi) * v1[0] - np.sin(phi) * v1[1]
    u[1] = np.sin(phi) * v1[0] + np.cos(phi) * v1[1]

    # print(np.round(u,0))

    print(Vector.angle(u, v))

def test2():
    naca = NacaProfile('6424', 100)
    naca.calculate_profile()
    naca.transform_points(10)

    profile = np.concatenate([naca.upper[::-1], naca.lower])
    
    points = PointMaker(1,-1,-1,1, 64,64).make()
    points = points.reshape(points.shape[0] * points.shape[1], points.shape[2])
    print(points.shape)

    tic = time.perf_counter()
    res = createDistField(profile, points)
    toc = time.perf_counter()

    print(res)
    print(f'time: {toc-tic} sec')

if __name__ =='__main__':
    test2()
