from src.NacaGenerator import NacaProfile
from src.tools import PointMaker, DistField
# from cython_stuff import shortest_distance_cy
import numpy as np
import time
import os

PATHS = {
    'profiles_list_file': './profiles_list',
    'profile_storage': '~/Documents/diplomka/NACA_data/'
}

AREA: float = 0.75
GRID_SIZE: int = 64
CAMBER_POINTS: int = 100


def create_input(profile: NacaProfile, number_camber_points: int, rotation: float, plot: bool = False) -> np.ndarray:

    points = PointMaker(AREA, -AREA, -AREA, AREA, GRID_SIZE, GRID_SIZE)

    dist_field = DistField(profile, points.make())

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
            yield prof_data[0], int(prof_data[1])

def main():

    for name, rot in profile_parameters():
        print(name, rot)

        naca = NacaProfile(name, CAMBER_POINTS)
        naca.calculate_profile()
        naca.transform_points(rot)

        dir_name: str = f'NACA_{name}_{rot}'

        if not os.path.isdir(f'{PATHS["profile_storage"]}/{dir_name}'):
            os.mkdir(os.path.join(f'{PATHS["profile_storage"]}',f'{dir_name}'))

        naca.to_dat(f'{PATHS["profile_storage"]}/{dir_name}')

    # inp = create_input(name, CAMBER_POINTS, rot, plot=True)
    # print(inp)

if __name__ == '__main__':
    main()