# Run this cell first to apply the style to the notebook
from IPython.core.display import HTML
css_file = './31380.css'
HTML(open(css_file, "r").read())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
%matplotlib inline

# create the dataframe and define a reduced version with 5000 randomized 
# observations of project goal and the total amount pledged
raw_data = pd.read_csv('ks-projects-201801.csv')
cols = ['goal', 'pledged']
df = pd.DataFrame(raw_data, columns = cols)

# 5000 random integers
index = np.random.randint(1,len(df),5000)
X = df.iloc[index,:]

# compute (reduced) dataframe dimensions (X = N x M)
N = len(df['goal'])
M = len(cols)

# plot a simple relationship
fig, ax = plt.subplots(figsize=(5,5))
ax.set(title='5000 projects up to 100k USD',
      xlabel='Project goal [USD]', ylabel='Money pledged [USD]',
      xlim=[0, 100000], ylim=[0, 100000])
ax.scatter(df_training['goal'], df_training['pledged'], alpha=0.5)
ax.plot(df_training['goal'], df_training['goal'], color='black', linewidth=1, label='Threshold')
ax.legend()
# idea: split dataset
success_mask = [df_training['pledged'] >= df_training['goal']]
print(success_mask[0:1])
# 'succeeded'-color blue, else 'failed'-red

# Subtract mean value from data
Y = X - X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
# print('U[0] = ', U[0], '\nS[0] =', S[0], '\nV[0] =', V[0])

# Compute variance explained by principal components
rho = (S**2) / (S**2).sum()

# Plot variance explained
fig, ax = plt.subplots(ncols=2)
ax1.plot(range(1,len(rho)+1),rho,'o-')
ax1.set(title='Variance explained by principal components', 
        xlabel='Principal component', ylabel='Variance explained')
