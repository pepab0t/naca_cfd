from subprocess import call
from src.NacaGenerator import NacaProfile
from src.tools import PointMaker, DistField
# from cython_stuff import shortest_distance_cy
import numpy as np
import time
import os
from contextlib import contextmanager

PATHS = {
    'profiles_list_file': './profiles_list',
    'profile_storage': '/home/cernikjo/Documents/diplomka/NACA_data/',
    'c2d_path': '/home/cernikjo/Construct2D_2.1.4/construct2d',
    'c2d_control': './src/c2d_control' 
}

AREA: float = 0.75
GRID_SIZE: int = 64
CAMBER_POINTS: int = 100

@contextmanager
def path_manager(path: str):
    previous_path: str = os.getcwd()

    os.chdir(path)
    yield 
    os.chdir(previous_path)

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

def call_c2d(source_path: str) -> None:
    '''Calls construct2d'''
    with open(PATHS['c2d_control']) as f:
        instructions: str = f.read()

    instructions = instructions.replace('#profile_path#', source_path+f'/{source_path.split("/")[-1]}.dat')
    with path_manager(source_path):
        os.system(f'echo \"{instructions}\" | {PATHS["c2d_path"]}')

def main():
    print(os.path.exists(PATHS['profile_storage']))
    for name, rot in profile_parameters():
        print(name, rot)

        naca = NacaProfile(name, CAMBER_POINTS)
        naca.calculate_profile()
        naca.transform_points(rot)

        dir_name: str = f'NACA_{name}_{rot}'

        if not os.path.isdir(f'{PATHS["profile_storage"]}/{dir_name}'):
            folder = os.path.join(f'{PATHS["profile_storage"]}',f'{dir_name}')
            os.mkdir(folder)

        naca.to_dat(f'{PATHS["profile_storage"]}/{dir_name}')



    # inp = create_input(name, CAMBER_POINTS, rot, plot=True)
    # print(inp)

if __name__ == '__main__':
    # main()
    call_c2d(f"{PATHS['profile_storage']}/NACA_0012_0")