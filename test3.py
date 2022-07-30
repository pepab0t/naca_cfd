import profile

import matplotlib.pyplot as plt

from MyCython.entities import Line
from src.NacaGenerator import NacaProfile


def main():
    profile = NacaProfile('6424', 100)
    profile.calculate_profile()

    data = profile.to_array()

    # plt.plot(data[:,0], data[:,1], 'o')
    # plt.axis('equal')
    # plt.show()

    print(data[0,:])
    line = Line(data[0,:], data[0,:])
    print(line)

if __name__ == '__main__':
    main()
