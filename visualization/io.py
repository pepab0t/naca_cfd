import matplotlib.pyplot as plt
import numpy as np

path = lambda x: f'../NACA_data/{x}'

def make_grid(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shp = int(np.sqrt(len(xyz)))
    
    X = xyz[:,0].reshape(shp, shp)
    Y = xyz[:,1].reshape(shp, shp)
    Z = xyz[:,2].reshape(shp, shp)

    return X, Y, Z

def plot_fields(inp: np.ndarray, out: np.ndarray) -> None:
    _, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))  # type: ignore

    x1,y1,z1 = make_grid(inp)
    ax1.contourf(x1, y1, z1, 100, cmap='coolwarm')
    ax1.axis('equal')

    x2,y2,z2 = make_grid(out)
    ax2.contourf(x2,y2,z2, 100, cmap='coolwarm')
    ax2.axis('equal')

    plt.show()

def main():
    prof_name: str = "NACA_0015_p1000"

    with open(f"{path(prof_name)}/input.npy", "rb") as f:
        inp: np.ndarray = np.load(f)
    
    with open(f"{path(prof_name)}/output.npy", "rb") as f:
        out: np.ndarray = np.load(f)

    # print(np.max(inp[:,:2] - out[:,:2]))

    plot_fields(inp, out)

if __name__ == "__main__":
    main()
