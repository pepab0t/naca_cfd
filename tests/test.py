import numpy as np
import matplotlib.pyplot as plt
import time
from createInput import Line, DistField, Vector
from MakePoints import PointMaker
from NacaGenerator import NacaProfile
import cython_stuff

def first():
    start = np.array([0,0.5])
    end = np.array([1,0])

    point = np.array([0.5, 0.25])

    l = Line(start, end)
    print(l.coeffs)

    dist_segment = l.distance_segment(point)
    # dist = l.distance(point)

    print(dist_segment)
    # print(dist)
    print(l.in_range(point))

    plt.plot(start[0], start[1],'o')
    plt.plot(end[0], end[1],'o')
    plt.plot([start[0], end[0]], [start[1], end[1]])
    plt.plot(point[0], point[1],'o')
    plt.axis('Equal')
    plt.show()

def test_outside():
    naca = NacaProfile('6424', 50)
    naca.calculate_profile()
    # naca.transform_points(10)

    points = PointMaker(1, -1, -1.5, 1.5, 4,4).make()

    dist_field = DistField(naca, points)

    point = np.array([0.4,0])
    print(naca.is_outside(point))

    # p = dist_field.evaluate()
    # print(p)


def third():
    origin = np.array([0,0])
    A = np.array([1,0])
    B = np.array([0,1])

    print(Vector.angle(A-origin, B-origin))

def fourth():
    n = NacaProfile('6424', 10)
    n.calculate_profile()
    print(n.lower)

def test_parallel():
    naca = NacaProfile('6424', 10)
    naca.calculate_profile()
    naca.transform_points(10)

    points = PointMaker(1, -1, -1.5, 1.5, 64,64)

    dist_field = DistField(naca, points.make())

    r = dist_field.evaluate_parallel()

def test_sequential():
    naca = NacaProfile('0012', 10)
    naca.calculate_profile()
    naca.transform_points(10)

    points = PointMaker(1, -1, -1.5, 1.5, 64,64)

    dist_field = DistField(naca, points.make())

    r = dist_field.evaluate()

def shortest_cython(n):
    naca = NacaProfile('6424', 100)
    naca.calculate_profile()
    naca.transform_points(10)

    p = np.array([0.5,0.5])

    tic = time.perf_counter()
    # for i in range(n):
    d1 = cython_stuff.shortest_distance_cy(naca.upper, naca.lower, p)
    toc = time.perf_counter()
    print(toc-tic)

    tic = time.perf_counter()
    # for i in range(n):
    d2 = cython_stuff.shortest_distance_py(naca.upper, naca.lower, p)
    toc = time.perf_counter()
    print(toc-tic, 'vanilla')

    return d1, d2    

def perf_time(foo):
    tic = time.perf_counter()
    out = foo()
    toc = time.perf_counter()

    print(toc - tic)
    return out

if __name__ == "__main__":
    # r1 = perf_time(test_parallel)
    # r2 = perf_time(test_sequential)
    # print(r2==r1)

    # test_outside()

    r = shortest_cython(1e5)
    print(r)