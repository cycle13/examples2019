# -*- coding: utf-8 -*-
from __future__ import print_function
__author__ = 'elesun'

# library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

out_dir = "output/"
# Get the data (csv file is hosted on the web)
url =  "https://python-graph-gallery.com/wp-content/uploads/volcano.csv"
data = pd.read_csv(url)
# data_dir = "data/volcano.csv"
# data = pd.read_csv(data_dir)

# Transform it to a long format
df=data.unstack().reset_index()
df.columns=["X","Y","Z"]


# And transform the old column name in something numeric
df["X"]=pd.Categorical(df["X"])
df["X"]=df["X"].cat.codes


# We are going to do 20 plots, for 20 different angles
for angle in range(70,210,2):
# Make the plot
    fig = plt.figure()
    ax = fig.gca(projection= "3d" )
    ax.plot_trisurf(df["Y"], df["X"], df["Z"], cmap=plt.cm.viridis, linewidth=0.2)

    ax.view_init(30,angle)

    filename= out_dir + str(angle)+ ".png"
    plt.savefig(filename, dpi=96)
    plt.gca()
    plt.close()
