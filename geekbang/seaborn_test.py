import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
df0 = pd.read_csv("D:/tensorflow/37/geekbang/data0.csv", names=['square', 'price'])
aa = sns.lmplot('square','price',df0,  fit_reg=True)