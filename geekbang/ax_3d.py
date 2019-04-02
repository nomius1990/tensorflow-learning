from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('data1.csv', names=['square', 'bedrooms', 'price'])
# temphead = df1.head()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')

i = ax.scatter3D(df1['square'], df1['bedrooms'], df1['price'], c=df1['price'], cmap='Greens')