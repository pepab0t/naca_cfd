import concurrent.futures
import os
import re
import time
from contextlib import contextmanager

# from cython_stuff import shortest_distance_cy
import numpy as np

from src.NacaGenerator import NacaProfile
from src.tools import DistField, PointMaker, timer

PATHS = {
    'profiles_list_file': './profiles_list',
    'profile_storage': '/home/cernikjo/Documents/diplomka/NACA_data/',
    'c2d_path': '/home/cernikjo/Construct2D_2.1.4/construct2d',
    'c2d_control': './src/c2d_control',
    'project': '../test'
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

def create_input(profile: NacaProfile, plot: bool = False) -> np.ndarray:

    maker = PointMaker(AREA, -AREA, -0.5, 1.5, GRID_SIZE, GRID_SIZE)
    points: np.ndarray = maker.make()

    maker.to_probes(points, PATHS['project'], repr=False)
    dist_field = DistField(profile, points)
    
    p2 = dist_field.evaluate_parallel()
    if plot:
        dist_field.plot_field(p2, show=True)

    return p2

# this is generator
def profile_parameters():
    with open(PATHS['profiles_list_file']) as f:
        for line in f.readlines():
            prof_data = line.strip().split(' ')
            if len(prof_data) != 2:
                raise ValueError('Bad input format (must be: name rotation)')
            yield prof_data[0], float(prof_data[1])

def call_c2d(source_path: str) -> None:
    '''Calls construct2d'''

    with open(PATHS['c2d_control']) as f:
        instructions: str = f.read()

    fname: str = source_path.split("/")[-1]

    instructions = instructions.replace('#profile_path#', f'{source_path}/{fname}.dat')
    c: int = 0
    with path_manager(source_path):
        while not os.path.isfile(f'{fname}.p3d'):
            os.system(f'echo \"{instructions}\" | {PATHS["c2d_path"]}')
            c += 1

        # remove thrash
        for filename in os.listdir():
            if re.search(r'(.*\.nmf|.*stats\.p3d)', filename):
                os.remove(filename)
                print(f'removed {filename}')
    print(f'construct2d runs: {c}')

# def array_to_file(array: np.ndarray, fname: str, directory: str):

def make_item(name: str, rot: float, plot: bool = False) -> None:
    print(name, rot)

    naca = NacaProfile(name, CAMBER_POINTS)
    naca.calculate_profile()
    naca.transform_points(rot)

    dir_name: str = f'NACA_{name}_{rot}'

    if not os.path.isdir(f'{PATHS["profile_storage"]}/{dir_name}'):
        folder = os.path.join(f'{PATHS["profile_storage"]}',f'{dir_name}')
        os.mkdir(folder)

    naca.to_dat(f'{PATHS["profile_storage"]}/{dir_name}')

    call_c2d(f"{PATHS['profile_storage']}/{dir_name}")

    dist_field = create_input(naca, plot)

    with open(f"{PATHS['profile_storage']}/{dir_name}/input.npy", 'wb') as f:
        np.save(f, dist_field)

@timer
def main(parallel: bool = False, plot:bool = False):

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            [executor.submit(make_item, name, rot, plot) for name, rot in profile_parameters()]
    else:
        for name, rot in profile_parameters():
            make_item(name, rot, plot)

if __name__ == '__main__':
    main(parallel=False, plot=True)

    # a = np.load(f"{PATHS['profile_storage']}/NACA_0012_0/input.npy")
    # print(a)
