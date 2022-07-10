import pandas as pd
import matplotlib.pyplot as plt

fpath = '../NacaAirfoil/postProcessing/surfaces/500/p_airfoil.raw'

df = pd.read_csv(fpath, sep=' ', skiprows=2, header=None)
df.columns = ['x', 'y', 'z', 'p']

# print(df['y'].max())

# plt.scatter(df['x'], df['z'])
plt.plot(df['x'], df['p'])
# plt.axis('equal')
plt.show()