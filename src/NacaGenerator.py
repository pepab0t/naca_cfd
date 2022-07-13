import math
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from src.entities import Line, Vector
except ModuleNotFoundError:
    from entities import Line, Vector

class VVector:
    def __init__(self, x, y, normalized=False):
        if normalized:
            norm = np.sqrt(x**2 + y**2)
        else:
            norm = 1
        self.x = x / norm
        self.y = y / norm

    def turn(self):
        return VVector(-self.x, -self.y)

    def to_tuple(self):
        return (self.x, self.y)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return VVector(self.x * other, self.y * other)

    def __add__(self, other):
        if isinstance(other, VVector):
            return VVector(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple) and len(other) == 2 and all([isinstance(x,float) for x in other]):
            return VVector(self.x + other[0], self.y + other[1])


class NacaProfile():
    X_OFF = 0

    def __init__(self, label: str, n: int):
        self.name = label
        self.rotation: float = 0
        self.n = n
        self.xc = np.zeros(self.n, dtype=float)
        self.yc = np.zeros(self.n, dtype=float)
        self.set_parameters(label)

    def set_parameters(self, number: str):
        # number = str(number)
        self.M = int(number[0])/100
        self.P = int(number[1])/10
        self.T = int(number[2:])/100

    @staticmethod
    def a_constants(trailling_edge='closed') -> tuple:
        if trailling_edge == 'closed':
            a4 = -0.1036
        elif trailling_edge == 'opened':
            a4 = -0.1015
        else:
            raise Exception('Unknows option, trailling_edge must be: opened, closed')

        return 0.2969, -0.126, -0.3516, 0.2843, a4

    def testing_profile(self):
        self.upper = np.array([
            [0,0],
            [0.3,0.5],
            [0.7,0.5],
            [1,0]
        ])

        self.lower = np.array([
            [0.3,-0.5],
            [0.7,-0.5]
        ])

    def calculate_profile(self, trailling_edge='opened'):
        '''trilling_edge: opened or closed'''

        yt = np.zeros(self.n, dtype=float)
        dyc_dx = np.zeros(self.n, dtype=float)
        theta = np.zeros(self.n, dtype=float)
        self.upper = np.zeros((self.n, 2), dtype=float)
        self.lower = np.zeros((self.n-1, 2), dtype=float)

        self.xc =  (1-np.cos(math.pi/(self.n-1) * np.arange(self.n)))/2 #+ self.X_OFF

        x1 = self.xc[self.xc <= self.P]
        x2 = self.xc[self.xc > self.P]

        a0, a1, a2, a3, a4 = self.a_constants(trailling_edge)

        if self.P != 0:
            self.yc[:len(x1)] = self.M/(self.P )**2 * (2*(self.P )* (x1) - (x1)**2)
            self.yc[len(x1):] = self.M/(1-(self.P ))**2 * (1 - 2*(self.P ) +2*(self.P )* (x2) - (x2)**2)

            dyc_dx[:len(x1)] = 2*self.M/(self.P )**2 * ((self.P ) - x1)
            dyc_dx[len(x1):] = 2*self.M / (1 - (self.P ))**2 * ((self.P ) - x2)
        else:
            self.yc[:] = 0
            dyc_dx[:] = 0

        yt = self.T / 0.2 * (a0*(self.xc)**0.5 + a1*(self.xc) + a2*(self.xc)**2 + a3*(self.xc)**3 + a4*(self.xc)**4)
        theta = np.arctan(dyc_dx)


        # for i in range(len(yc)):
        self.upper[:, 0] = self.xc - yt * np.sin(theta) - 0.5
        self.upper[:, 1] = self.yc + yt * np.cos(theta)
        self.lower[:, 0] = self.xc[1:] + yt[1:] * np.sin(theta[1:]) - 0.5
        self.lower[:, 1] = self.yc[1:] - yt[1:] * np.cos(theta[1:])

        # self.tranform_points()

    def transform_points(self, phi: float):
        '''Phi in degrees'''
        if not self.is_ready():
            raise Exception("Data is empty. Call self.calculate_profile() first.")
        
        self.rotation = phi
        phi = -phi/180 * np.pi

        upper_old = self.upper.copy()
        lower_old = self.lower.copy()

        self.upper[:,0] = upper_old[:,0] * np.cos(phi) - upper_old[:,1] * np.sin(phi)
        self.upper[:,1] = upper_old[:,0] * np.sin(phi) + upper_old[:,1] * np.cos(phi)
        
        self.lower[:,0] = lower_old[:,0] * np.cos(phi) - lower_old[:,1] * np.sin(phi)
        self.lower[:,1] = lower_old[:,0] * np.sin(phi) + lower_old[:,1] * np.cos(phi)


    def is_ready(self):
        return (self.upper!=0).any() and (self.lower!=0).any()

    def is_outside(self, point: np.ndarray) -> bool:
        Line._validate_entity(point)

        sum_angle = 0

        profile_points = np.concatenate([self.upper[::-1], self.lower])


        for i in range(1,len(profile_points)):
            a = profile_points[i-1, :] - point
            b = profile_points[i, :] - point
            # print(profile_points[i-1, :], profile_points[i, :])
            alpha = np.arcsin( Vector.vector_mul(a, b) / (Vector.magnitude(a) * Vector.magnitude(b))) * 180/np.pi
            # print(alpha)
            # self.disp(camber=False, show=False)
            # plt.plot([point[0], point[0]+a[0]], [point[1], point[1]+a[1]], 'r')
            # plt.plot([point[0], point[0]+b[0]], [point[1], point[1]+b[1]], 'b')
            # plt.show()
            sum_angle += alpha

        return sum_angle < 180

    def to_geo(self, profile_size=0.1, area_size=1, area=[(-5, 5), (7, 5), (7, -5), (-5, -5)]):
        lines = range(1, self.n)
        points = 1
        lines = 1
        lineloops = 1
        
        with open(f'NACA_{self.name}.geo', 'w+') as f:
            # area points
            for p in area:
                f.write(f'Point({points}) = ' + '{' + f'{p[0]}, {p[1]}, 0, {area_size}' + '};\n')
                points += 1
            
            # area lines
            for linepoint in range(1, len(area)+1):
                if linepoint == 4:
                    f.write(f'Line({lines}) = ' + '{' + f'{linepoint}, {1}' +'};\n')
                else:
                    f.write(f'Line({lines}) = ' + '{' + f'{linepoint}, {linepoint+1}' +'};\n')
                lines += 1

            # area loop
            f.write(f'Curve Loop({lineloops}) = {{1, 2, 3, 4}};\n')
            lineloops += 1

            # profile points
            profile_start_point = points
            for row in self.upper:
                f.write(f'Point({points}) = ' + '{' + f'{row[0]}, {row[1]}, 0, {profile_size}' + '};\n')
                points += 1
            f.write('\n')

            for row in self.lower[-2::-1]:
                f.write(f'Point({points}) = ' + '{' + f'{row[0]}, {row[1]}, 0, {profile_size}' + '};\n')
                points += 1

            # profile lines
            profile_start_line = lines
            for i in range(profile_start_point, points):
                if i == points - 1:
                    f.write(f'Line({lines}) = ' + '{' + f'{i}, {profile_start_point}' +'};\n')
                else:
                    f.write(f'Line({lines}) = ' + '{' + f'{i}, {i+1}' +'};\n')
                lines += 1

            # profile loop
            prof_edges = map(lambda x: str(x), list(range(profile_start_line, lines)))
            prof_edges = '{' + ', '.join(prof_edges) + '}'
            f.write(f'Curve Loop({lineloops}) = {prof_edges};\n')
            lineloops += 1

            # surface
            f.write("Plane Surface(1) = {1,2};\n")

            f.write('Field[1] = BoundaryLayer;\n')
            f.write(f'Field[1].EdgesList = {prof_edges};\n')
            f.write('Field[1].hwall_n = 0.00001;\n')
            f.write('Field[1].thickness = 0.1;\n')
            f.write('Field[1].ratio = 1.1;\n')
            f.write('BoundaryLayer Field = 1;\n')

            f.write('Recombine Surface{1};\n') ############################################################################
            f.write('MeshAlgorithm Surface{1} = 8;\n')
            # f.write('Extrude {0, 0, 1} {\n')
            # f.write('Surface{1};\n')
            # f.write('Layers{1};\n')
            # f.write('Recombine;}\n')

    def to_dat(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Invaldi path: {path}')
        with open(f'{path}/NACA_{self.name}_{self.rotation}.dat', 'w+') as f:
            f.write(f'NACA {self.name}\n')

            for row in self.upper[::-1]:
                f.write(f' {row[0]} {row[1]}\n')
            
            for row in self.lower:
                f.write(f' {row[0]} {row[1]}\n')


    def __repr__(self):
        out = f'NACA {self.name}\n'

        out += 'CAMBER:\n'
        for i in range(self.n):
            line = f' x = {self.xc[i]}, y = {self.yc[i]}\n'
            out += line

        out += '\nUPPER:\n'

        for row in self.upper:
            line = f' x = {row[0]}, y = {row[1]}\n'
            out += line
        
        out += '\nLOWER:\n'

        for row in self.lower:
            line = f' x = {row[0]}, y = {row[1]}\n'
            out += line

        return out

    def disp(self, camber:bool=True, show: bool=False):
        if camber:
            plt.plot(self.xc, self.yc, color='black', ls='--', lw=1)
        plt.plot(self.upper[:, 0], self.upper[:, 1], 'ok')
        plt.plot(self.lower[:, 0], self.lower[:, 1], 'ok')
        plt.axis('equal')
        plt.grid(True, color='#d9d9d9', linestyle='--', alpha=0.6)
        if show:
            plt.show()

def main():
    #, area=[(-2, 2), (3,2), (3,-2), (-2,-2)]
    p = NacaProfile('6000', 100)
    p.calculate_profile('opened')
    p.transform_points(20)
    # print(p)
    p.disp()
    p.to_dat()
    # p.to_geo(0.05, 0.5)

if __name__ == '__main__':
    main()
