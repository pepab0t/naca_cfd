from src.NacaGenerator import NacaProfile
from src.tools import PointMaker, DistField
# from cython_stuff import shortest_distance_cy
import numpy as np
import time
import os

PATHS = {
    'profiles_list_file': './profiles_list',
    'profile_storage': '~/Documents/diplomka/NACA_data'
}

AREA: float = 0.75
GRID_SIZE: int = 64
CAMBER_POINTS: int = 100


def create_input(profile_label: str, number_camber_points: int, rotation: float, plot: bool = False) -> np.ndarray:
    naca = NacaProfile(profile_label, number_camber_points)
    naca.calculate_profile()

    naca.transform_points(rotation)

    points = PointMaker(AREA, -AREA, -AREA, AREA, GRID_SIZE, GRID_SIZE)

    dist_field = DistField(naca, points.make())

    tic = time.perf_counter()
    p2 = dist_field.evaluate_parallel()
    toc = time.perf_counter()
    print(toc-tic)

    if plot:
        dist_field.plot_input(p2, show=True)

    return p2

def profile_parameters():
    with open(PATHS['profiles_list_file']) as f:
        for line in f.readlines():
            prof_data = line.strip().split(' ')
            if len(prof_data) != 2:
                raise ValueError('Bad input format (must be: name rotation)')
            yield prof_data[0], prof_data[1]

def main():

    for name, rot in profile_parameters():
        print(name, rot)

    # inp = create_input('6424', CAMBER_POINTS, 0, plot=True)
    # print(inp)

if __name__ == '__main__':
    main()