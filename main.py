import concurrent.futures
import os
import re
import shutil

# from cython_stuff import shortest_distance_cy
import numpy as np

from src.NacaGenerator import NacaProfile
from src.probes import probes_to_array
from src.tools import (DistField, PointMaker, directory_name, path_manager,
                       timer)

PATHS = {
    'profiles_list_file': './profiles_list',
    'profile_storage': '/home/cernikjo/Documents/diplomka/NACA_data/',
    'c2d_path': '/home/cernikjo/Construct2D_2.1.4/construct2d',
    'c2d_control': './src/c2d_control',
    'project': '../naca_clean',
    'default_path': '../' 
}

AREA: tuple[float, float, float, float] = (1, -1, -0.5, 1.5)
GRID_SIZE: int = 64
CAMBER_POINTS: int = 100



def create_dist_field(profile: NacaProfile, plot: bool = False) -> np.ndarray:

    maker = PointMaker(*AREA, GRID_SIZE, GRID_SIZE)
    points: np.ndarray = maker.make()

    maker.to_probes(points, PATHS['project'], repr=False)
    dist_field = DistField(profile, points)
    
    p: np.ndarray = dist_field.evaluate_parallel()
    
    if plot:
        dist_field.plot_field(p, show=True)

    return p

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

def make_item(name: str, rot: float, plot: bool = False) -> None:
    print(name, rot)

    naca = NacaProfile(name, CAMBER_POINTS)
    naca.calculate_profile()

    dir_name: str = directory_name(name, rot)

    if not os.path.isdir(f'{PATHS["profile_storage"]}/{dir_name}'):
        folder = os.path.join(f'{PATHS["profile_storage"]}',f'{dir_name}')
        os.mkdir(folder)

    naca.to_dat(f'{PATHS["profile_storage"]}/{dir_name}')

    call_c2d(f"{PATHS['profile_storage']}/{dir_name}")

    naca.transform_points(rot)
    dist_field = create_dist_field(naca, plot)

    with open(f"{PATHS['profile_storage']}/{dir_name}/input.npy", 'wb') as f:
        np.save(f, dist_field)

def run_foam(name: str, rot: float):

    dir_name: str = directory_name(name, rot)
    
    shutil.copytree(f"{PATHS['default_path']}/naca_clean", f"{PATHS['profile_storage']}/{dir_name}/compute", dirs_exist_ok=True)

    shutil.move(f"{PATHS['profile_storage']}/{dir_name}/{dir_name}.p3d", f"{PATHS['profile_storage']}/{dir_name}/compute/mesh.p3d")

    with path_manager(f"{PATHS['profile_storage']}/{dir_name}/compute"):
        os.system(f'sh commands.sh mesh.p3d {rot}')
        os.system('simpleFoam')
        shutil.move('postProcessing/probes/0/p', '../p')
        # shutil.rmtree(path)
        
    with path_manager(f"{PATHS['profile_storage']}/{dir_name}"):
        with open(f"output.npy", "wb") as f:
            np.save(f, probes_to_array())
            

@timer
def main(foam: bool = False, plot:bool = False):

    for name, rot in profile_parameters():
        make_item(name, rot, plot)

    if foam:
        for name, rot in profile_parameters():
            run_foam(name, rot)

        

if __name__ == '__main__':
    main(True, plot=True)

