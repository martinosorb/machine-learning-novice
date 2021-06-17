---
title: "Dimensionality Reduction"
teaching: 0
exercises: 0
questions:
- "How can we perform unsupervised learning with dimensionality reduction techniques such as Principle Component Analyis (PCA), 
Uniform Manifold Approximation and Projection  (UMAP), t-distributed Stochastic Neighbor Embedding (t-SNE) and Higher Order Singular Value Decomposition (HOSVD) ?"
objectives:
- "Recall that most data is inherently multidimensional"
- "Understand that reducing the number of dimensions can simplify modelling and allow classifications to be performed."
- "Understand that dimensionality reduction is helpful in performing data visualization and interpretation"
- "Apply PCA, t-SNE, UMAP and HOSVD with Scikit Learn to an example dataset."
- "Evaluate the relative peformance of PCA, t-SNE, UMAP and HOSVD."
keypoints:
- "PCA is a linear dimensionality reduction technique for tabular data"
- "t-SNE is another dimensionality reduction technique for tabular data that is more general than PCA"
- "UMAP is another dimensionality reduction technique for tabular that allows for nonlinear embeddings"
- "HOSVD is a linear dimensionality reduction technique for general tensors"

---

# Dimensionality Reduction

Dimensionality reduction is the process of using a subset of the coordinates, 
which may be transformed, of the dataset to capture the variation in features 
of the data set.  It can be a helpful pre-processing step before doing other 
operations on the data, such as classification, regression or visualization.

## Dimensionality Reduction with Scikit-learn

The script below compares PCA, t-SNE, UMAP and HOSVD for the MNIST digits dataset.

First setup our environment and load the MNIST digits dataset which will be used 
as our initial example.
~~~
# https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py
# https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
# https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
# https://umap.scikit-tda.org/basic_usage.html#digits-data
# https://towardsdatascience.com/how-to-program-umap-from-scratch-e6eff67f55fe
# https://dev.to/juliasilge/pca-and-umap-with-cocktail-recipes-4980
# https://pair-code.github.io/understanding-umap/
# https://pypi.org/project/umap-learn/
# https://jlmelville.github.io/uwot/abparams.html

# SETUP
# pip3 install scikit-learn
# pip3 install umap-learn

import numpy as np
import matplotlib.pyplot as plt
import umap

from sklearn import decomposition
from sklearn import datasets
from sklearn import manifold

digits = datasets.load_digits()

# Examine the dataset

print(digits.data)

print(digits.target)

X = digits.data
y = digits.target

~~~
{: .python}

### Principle Component Analysis (PCA)

PCA is a technique that does rotations of data in a two dimensional
array to decompose the array into combinations vectors that are orthogonal
and can be ordered according to the amount of information they carry.

~~~
# PCA
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.nipy_spectral, 
        edgecolor='k',label=y)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.savefig("pca.svg")
~~~
{: .python}

![Reduction using PCA](../fig/pca.svg)

### t-distributed Stochastic Neighbor Embedding (t-SNE)

~~~
# t-SNE embedding
tsne = manifold.TSNE(n_components=2, init='pca',
        random_state = 0)
X_tsne = tsne.fit_transform(X)
fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.nipy_spectral,
        edgecolor='k',label=y)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.savefig("tsne.svg")
~~~
{: .python}

![Reduction using t-SNE](../fig/tsne.svg)

### Uniform Manifold Approximation (UMAP)

~~~
# umap
reducer= umap.UMAP(n_components=2, n_neighbors=15,
   random_state=42, transform_seed=42, verbose=False)
reducer.fit(X)

X_umap = reducer.transform(X)

fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=plt.cm.nipy_spectral,
        edgecolor='k',label=y)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.savefig("umap.svg")
~~~
{: .python}

![Reduction using umap](../fig/umap.svg)

> # Exercise: Working in three dimensions
> The above example has considered only two dimensions since humans
> can visualize two dimensions very well. However, there can be cases
> where a dataset requires more than two dimensions to be appropriately
> decomposed. Modify the above programs to use three dimensions and 
> create appropriate plots.
> Do three dimensions allow one to better distinguish between the digits?
>
> > ## Solution
> > ~~~
> > from mpl_toolkits.mplot3d import Axes3D
> > # PCA
> > pca = decomposition.PCA(n_components=3)
> > pca.fit(X)
> > X_pca = pca.transform(X)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y,
> >           cmap=plt.cm.nipy_spectral, s=9, lw=0)
> > plt.savefig("pca_3d.svg")
> > ~~~
> > {. :python}
> >
> > ![Reduction to 3 components using pca](../fig/pca_3d.svg)
> >
> > ~~~
> > # t-SNE embedding
> > tsne = manifold.TSNE(n_components=3, init='pca',
> >         random_state = 0)
> > X_tsne = tsne.fit_transform(X)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y,
> >           cmap=plt.cm.nipy_spectral, s=9, lw=0)
> > plt.savefig("tsne_3d.svg")
> > ~~~
> > {. :python}
> >
> > ![Reduction to 3 components using tsne](../fig/tsne_3d.svg)
> >
> > ~~~
> > # umap
> > reducer= umap.UMAP(n_components=3, n_neighbors=15,
> >    random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(digits.data)
> > X_umap = reducer.transform(digits.data)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=y,
> >           cmap=plt.cm.nipy_spectral, s=9, lw=0)
> > plt.savefig("umap_3d.svg")
> > ~~~
> > {. :python}
> >
> > ![Reduction to 3 components using umap](../fig/umap_3d.svg)
> >
>
> # Exercise: Parameters
>
> Look up parameters that can be changed in PCA, t-SNE and UMAP
> and experiment with these. How do they change your resulting
> plots?  Might the choice of parameters lead you to make different
> conclusions about your data?
>
> # Exercise: Other Datasets
>
> ## Fashion MNIST
>
> Use PCA, t-SNE and UMAP to try to classify the
> [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
>
> > ### Solution
> >
> > The Fashion MNIST dataset can be downloaded using [pymde](https://github.com/cvxgrp/pymde/blob/main/examples/fashion_mnist.ipynb).
> >
> > ~~~
> > import trimap
> > from sklearn.datasets import load_digits
> > import matplotlib.pyplot as plt
> > import numpy as np
> > import pymde
> > import pacmap
> > import umap
> > from sklearn import decomposition
> > from sklearn import manifold
> > ### Get data
> > f_mnist = pymde.datasets.FashionMNIST()
> > X = np.array(f_mnist.data,dtype=np.float64)
> > y = f_mnist.attributes['class']
> >
> > ## PCA
> > pca = decomposition.PCA(n_components=2)
> > pca.fit(X)
> > X_pca = pca.transform(X)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
> >             edgecolor='k',label=y)
> > plt.savefig("f_mnist_pca.svg")
> > ~~~
> > {. :python}
> > ![Reduction to 2 dimensions of F MNIST using PCA](../fig/f_mnist_pca.svg)
> > ~~~
> > ### tsne
> > #### This can take some time to complete
> > tsne = manifold.TSNE(n_components=2, init='pca',
> >        random_state = 0)
> > X_tsne = tsne.fit_transform(X)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("f_mnist_tsne.svg")
> > ~~~
> > {. :python}
> > ![Reduction to 2 dimensions of F MNIST using tsne](../fig/f_mnist_tsne.svg)
> > ~~~
> > ### umap
> > #### This can also be run in a supervised learning mode
> > #### see https://umap-learn.readthedocs.io/en/latest/supervised.html#umap-on-fashion-mnist
> > reducer= umap.UMAP(n_components=2, n_neighbors=5,
> >   random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(X)
> >
> > X_umap = reducer.transform(X)
> >
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("f_mnist_umap.svg")
> > ~~~
> > {. :python}
> >  ![Reduction to 2 dimensions of F MNIST using UMAP](../fig/f_mnist_umap.svg)
> > ~~~
> > ### trimap
> > f_mnist_trimap = trimap.TRIMAP().fit_transform(X)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(f_mnist_trimap[:, 0], f_mnist_trimap[:, 1], c=y, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("f_mnist_trimap.svg")
> > ~~~
> > {. :python}
> >  ![Reduction to 2 dimensions of F MNIST using TriMAP](../fig/f_mnist_trimap.svg)
> > ~~~
> > ### PACMAP
> > f_mnist_pacmap = pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(X, init="pca")
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(f_mnist_pacmap[:, 0], f_mnist_pacmap[:, 1], c=y, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("f_mnist_pacmap.svg")
> > ~~~
> > {. :python}
> >  ![Reduction to 2 dimensions of F MNIST using PaCMAP](../fig/f_mnist_pacmap.svg)
> > ~~~
> > ### MDE
> > f_mnist_mde = pymde.preserve_neighbors(X,verbose=True).embed(verbose=True)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(f_mnist_mde[:, 0], f_mnist_mde[:, 1], c=y, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("f_mnist_pymde.svg")
> > ~~~
> > {. :python}
> >  ![Reduction to 2 dimensions of F MNIST using MDE](../fig/f_mnist_mde.svg)
> {. :solution}
>
> ## Nonlinear Manifolds
>
> The AstroML package has an [example](https://www.astroml.org/book_figures/chapter7/fig_S_manifold_PCA.html#book-fig-chapter7-fig-s-manifold-pca) 
> where principle components analysis can destroy nonlinear 
> structure, and shows that other methods can be better.
> It uses an S-curve as a dataset.
> 
> ~~~ 
> num_points = 1500
> X, color = datasets.make_s_curve(num_points, random_state=0)
>
> # change the proportions to emphasize the weakness of PCA
> X[:, 1] -= 1
> X[:, 1] *= 1.5
> X[:, 2] *= 0.5
>
> from mpl_toolkits.mplot3d import Axes3D
>
> ax = fig.add_subplot(projection='3d')
> ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,
>            cmap=plt.cm.nipy_spectral, s=9, lw=0)
> plt.savefig("s_curve.svg")
> ~~~
> {: .python}
> 
> ![S shaped two dimensional manifold in three dimensional space](../fig/s_curve.svg)
>
> Compare the two dimensional projections generated by PCA, t-SNE and UMAP
> for this data. What happens if you comment out the adjustments on
> `X[:, 1] -= 1`, `X[:, 1] *= 1.5` and `X[:, 2] *= 0.5`.
>
> > ~~~
> > # PCA
> > pca = decomposition.PCA(n_components=2)
> > pca.fit(X)
> > X_pca = pca.transform(X)
> >
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color, 
> >             edgecolor='k',label=color)
> > plt.savefig("pca_s_curve.svg")
> > ~~~
> > {: .python}
> >
> > ![Reduction of nonlinear s curve manifold using PCA](../fig/pca_s_curve.svg)
> >
> > ### t-distributed Stochastic Neighbor Embedding (t-SNE)
> >
> > ~~~
> > # t-SNE embedding
> > tsne = manifold.TSNE(n_components=2, init='pca',
> >        random_state = 0)
> > X_tsne = tsne.fit_transform(X)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color,
> >        edgecolor='k',label=color)
> > plt.savefig("tsne_s_curve.svg")
> > ~~~
> > {: .python}
> >
> > ![Reduction of nonlinear s curve using t-SNE](../fig/tsne_s_curve.svg)
> >
> > ### Uniform Manifold Approximation (UMAP)
> >
> > ~~~
> > # umap
> > reducer= umap.UMAP(n_components=2, n_neighbors=15,
> >   random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(X)
> >
> > X_umap = reducer.transform(X)
> >
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(X_umap[:, 0], X_umap[:, 1], c=color,
> >        edgecolor='k',label=color)
> > plt.savefig("umap_s_curve.svg")
> > ~~~
> > {: .python}
> >
> > ![Reduction of nonlinear s curve using umap](../fig/umap_s_curve.svg)
> >
>
> ## Raman Spectra
> PCA can be used to characterize Raman spectra in cartilage samples to
> distinguish between different constituents.
> ~~~
> import pandas as pd
> import numpy as np
> import tensorly as tl
>
> specdata = pd.read_csv('chondro.txt',sep='\s+',header=None)
> specdata = specdata.interpolate(method='linear',limit_direction='forward')
> pca = decomposition.PCA(n_components=2)
> pca.fit(specdata.to_numpy())
> specdata_pca = pca.transform(specdata.to_numpy())
>
> fig = plt.figure(1, figsize=(4, 4))
> plt.clf()
> plt.scatter(specdata_pca[:, 0], specdata_pca[:, 1], edgecolor='k')
> plt.colorbar()
> plt.savefig("specdata_pca.png")
>
> # convert to tensor
> x = np.unique((specdata[[0]]).to_numpy())
> y = np.unique((specdata[[1]]).to_numpy())
> raman_shift = np.unique((specdata[[2]]).to_numpy())
> intensity = (specdata[[3]]).to_numpy()
> intensity = np.reshape(intensity, [ raman_shift.size, y.size, x.size])
>
> ~~~
> {: .python}
>
> Find areas in physics where spectroscopic data is also collected.
> Discuss how similar dimension reduction techniques may be useful when
> examining such data
>
> > ## Solution
> > [DESI (Dark Energy Spectroscopic Instrument)](https://www.desi.lbl.gov/) 
>
> ## Galaxies
>
> Classifying galaxies is also important. Example data is available on the
> [Galaxy zoo](https://data.galaxyzoo.org/) and 
> [Sloan Digital Sky Survey](https://www.sdss.org) webpages. These datasets 
> are quite large.  More managable datasets are the 
> [Kaggle galaxy zoo challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data),
> the [EFIGI catalogue of 4458 nearby galaxies with morphology](http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/532/A74),
> the [Zsolt Frei dataset](http://www.zsolt-frei.net/galaxy_catalog.html),
> the [Galaxy 10](https://github.com/henrysky/Galaxy10),
> and the [Galaxy 10 SDSS](https://astronn.readthedocs.io/en/latest/galaxy10sdss.html#download-galaxy10-sdss) datasets.
> Use PCA, t-SNE, UMAP and HOSVD to try and classify the galaxies in one of these datasets. 
>
> > ## Solution
> > The [Galaxy 10 SDSS](https://astronn.readthedocs.io/en/latest/galaxy10sdss.html#download-galaxy10-sdss) 
> > is about 200Mb, has been pre-processed and contains images which are useful for introducing
> > machine learning.
> > ~~~
> > # import everything we need first
> > import numpy as np
> > import h5py
> > import matplotlib.pyplot as plt
> > import pymde
> > import pacmap
> > import umap
> > import pymde
> > import trimap
> > from sklearn import decomposition
> > from sklearn import manifold
> > from sklearn.model_selection import train_test_split
> > import math
> > 
> > # To get the images and labels from file
> > with h5py.File('Galaxy10.h5', 'r') as F:
> >     images = np.array(F['images'])
> >    labels = np.array(F['ans'])
> > X_train = np.empty([math.floor(len(labels)/100),14283],dtype=np.float64)
> > y_train = np.empty([math.floor(len(labels)/100)],dtype=np.float64)
> > X_test = X_train
> > y_test = y_train
> > # Try to improve this by creating more balanced data subsets
> > # and using randomization
> > for i in range(math.floor(len(labels)/100)):
> >     X_train[i,:]=np.array(np.ndarray.flatten(images[i,:,:,:]),dtype=np.float64)
> >    y_train[i]=labels[i]
> >    X_test[i,:]=np.array(np.ndarray.flatten(images[i+math.floor(len(labels)/100),:,:,:]),dtype=np.float64)
> >    y_test[i]=labels[i+math.floor(len(labels)/100)]
> >
> > # Plot distribution
> > classes, frequency = np.unique(y_train,return_counts=True)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.bar(classes,frequency)
> > plt.xlabel('Class')
> > plt.ylabel('Frequency')
> > plt.title('Data Subset')
> > plt.savefig("galaxy10_subset.svg")
> >
> > ~~~
> > {: .python}
> > 
> > ![Class distribution in subset of training data for Galaxy10](../fig/galaxy10_subset.svg)
> >
> > #### 2D Embedding
> > ~~~
> > ### PCA
> > pca = decomposition.PCA(n_components=2)
> > pca.fit(X_train)
> > galaxy10_pca = pca.transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_pca[:, 0], galaxy10_pca[:, 1], c=y_train, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y_train)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_pca.svg")
> > ~~~
> > {: .python}
> >
> > ![PCA results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_pca.svg) 
> >
> > ~~~
> > ### trimap
> > galaxy10_trimap = trimap.TRIMAP().fit_transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_trimap[:, 0], galaxy10_trimap[:, 1], c=y_train, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y_train)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_trimap.svg")
> > ~~~
> > {: .python}
> >
> > ![PCA results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_trimap.svg)
> >
> > ~~~
> > ### MDE
> > galaxy10_mde = pymde.preserve_neighbors(X_train,embedding_dim=2,verbose=True).embed(verbose=True)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_mde[:, 0], galaxy10_mde[:, 1], c=y_train, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y_train)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_pymde.svg")
> > ~~~
> > {: .python}
> >
> > ![MDE results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_mde.svg)
> >
> > ~~~
> > ### PACMAP
> > galaxy10_pacmap = pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(X_train, init="pca")
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_pacmap[:, 0], galaxy10_pacmap[:, 1], c=y_train, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y_train)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_pacmap.svg")
> > ~~~
> >
> > ![PaCMAP results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_pacmap.svg)
> >
> > ~~~
> > ### UMAP
> > reducer= umap.UMAP(n_components=2, n_neighbors=5,
> >   random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(X_train)
> >
> > galaxy10_umap = reducer.transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_umap[:, 0], galaxy10_umap[:, 1], c=y_train, cmap=plt.cm.nipy_spectral,
> >        edgecolor='k',label=y_train)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_umap.svg")
> > ~~~
> >
> > ![UMAP results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_umap.svg)
> >
> > ~~~
> > ### UMAP - Supervised
> > reducer= umap.UMAP(n_components=2, n_neighbors=15,
> >   random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(X_train,y_train)
> >
> > galaxy10_umap_supervised = reducer.transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_umap_supervised[:, 0], galaxy10_umap_supervised[:, 1],
> >        c=y_train, cmap=plt.cm.nipy_spectral,edgecolor='k',label=y_train)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_umap_supervised.svg")
> > ~~~
> > {. :python}
> >
> > ![Supervised UMAP results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_umap_supervised.svg)
> >
> > ~~~
> > ### UMAP - Supervised prediction
> > galaxy10_umap_supervised_prediction = reducer.transform(X_test)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > plt.scatter(galaxy10_umap_supervised_prediction[:, 0], galaxy10_umap_supervised_prediction[:, 1],
> >        c=y_test, cmap=plt.cm.nipy_spectral,edgecolor='k',label=y_test)
> > plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_2D_umap_supervised_prediction.svg")
> > ~~~
> > {. :python}
> >
> > ![UMAP prediction results for Galaxy10SDSS in 2D](../fig/galaxy10_2D_umap_supervised_prediction.svg)
> >
> > ### 3D Embedding
> >
> > ~~~
> > ### PCA
> > pca = decomposition.PCA(n_components=3)
> > pca.fit(X_train)
> > galaxy10_pca = pca.transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_pca[:, 0], galaxy10_pca[:, 1],
> >               galaxy10_pca[:, 2], c=y_train,
> >               cmap=plt.cm.nipy_spectral, edgecolor='k',label=y_train)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_pca.svg")
> > ~~~
> > {. :python}
> >
> > ![PCA results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_pca.svg)
> >
> > ~~~
> > ### trimap
> > galaxy10_trimap = trimap.TRIMAP(n_dims=3).fit_transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_trimap[:, 0], galaxy10_trimap[:, 1],
> >                galaxy10_trimap[:, 2], c=y_train,
> >                cmap=plt.cm.nipy_spectral, edgecolor='k',label=y_train)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_trimap.svg")
> > ~~~
> > {. :python}
> >
> > ![TriMAP results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_trimap.svg)
> >
> > ~~~
> > ### MDE
> > galaxy10_mde = pymde.preserve_neighbors(X_train,embedding_dim=3,verbose=True).embed(verbose=True)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_mde[:, 0], galaxy10_mde[:, 1],
> >               galaxy10_mde[:, 2], c=y_train,
> >               cmap=plt.cm.nipy_spectral, edgecolor='k',label=y_train)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_pymde.svg")
> > ~~~
> > {. :python}
> >
> > ![MDE results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_mde.svg)
> >
> > ~~~
> > ### PaCMAP
> > galaxy10_pacmap = pacmap.PaCMAP(n_dims=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0).fit_transform(X_train, init="pca")
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_pacmap[:, 0], galaxy10_pacmap[:, 1],
> >               galaxy10_pacmap[:, 2], c=y_train,
> >               cmap=plt.cm.nipy_spectral, edgecolor='k',label=y_train)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_pacmap.svg")
> > ~~~
> > {. :python}
> >
> > ![PaCMAP results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_pacmap.svg)
> >
> > ~~~
> > ### UMAP
> > reducer= umap.UMAP(n_components=3, n_neighbors=5,
> >   random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(X_train)
> >
> > galaxy10_umap = reducer.transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_umap[:, 0], galaxy10_umap[:, 1],
> >              galaxy10_umap[:, 2], c=y_train,
> >              cmap=plt.cm.nipy_spectral,edgecolor='k',label=y_train)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_umap.svg")
> > ~~~
> > {. :python}
> >
> > ![UMAP results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_umap.svg)
> >
> > ~~~
> > ### UMAP - Supervised
> > reducer= umap.UMAP(n_components=3, n_neighbors=15,
> >   random_state=42, transform_seed=42, verbose=False)
> > reducer.fit(X_train,y_train)
> >
> > galaxy10_umap_supervised = reducer.transform(X_train)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_umap_supervised[:, 0], galaxy10_umap_supervised[:, 1],
> >              galaxy10_umap_supervised[:, 2],c=y_train,
> >              cmap=plt.cm.nipy_spectral,edgecolor='k',label=y_train)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_umap_supervised.svg")
> > ~~~
> > {. :python}
> >
> > ![Supervised UMAP results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_umap_supervised.svg)
> >
> > ~~~
> > ### UMAP - Supervised prediction
> >
> > galaxy10_umap_supervised_prediction = reducer.transform(X_test)
> > fig = plt.figure(1, figsize=(4, 4))
> > plt.clf()
> > ax = fig.add_subplot(projection='3d')
> > p = ax.scatter(galaxy10_umap_supervised_prediction[:, 0], galaxy10_umap_supervised_prediction[:, 1],
> >              galaxy10_umap_supervised_prediction[:, 2],c=y_test,
> >              cmap=plt.cm.nipy_spectral,edgecolor='k',label=y_test)
> > fig.colorbar(p,ax=ax,boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
> > plt.savefig("galaxy10_3D_umap_supervised_prediction.svg")
> > ~~~
> > {. :python}
> >
> > ![UMAP prediction results for Galaxy10SDSS in 3D](../fig/galaxy10_3D_umap_supervised_prediction.svg)
> >
>
> # Exercise: Other Algorithms
>
> There are other algorithms that can be used for doing dimensionality
> reduction, for example the Higher Order Singular Value Decomposition (HOSVD)
> Do an internet search for some of these and
> examine the example data that they are used on. Are there cases where they do 
> poorly? What level of care might you need to use before applying such methods
> for automation in critical scenarios?  What about for interactive data 
> exploration?

# References
- Connolly, Szalay, Bershady, Kinney and Calzetti, [Spectral classification of galaxies: an orthogonal approach](https://arxiv.org/abs/astro-ph/9411044)
- Connolly and Szalay, [A Robust Classification of Galaxy Spectra: Dealing with Noisy and Incomplete Data](https://arxiv.org/abs/astro-ph/9901300)
- Dielman, Willett and Dambre, [Rotation-invariant convolutional neural networks for galaxy morphology prediction](https://arxiv.org/abs/1503.07077)
- Selim, Keshk and Shourbygy, [Galaxy Image Classification using Non-Negative Matrix Factorization](https://www.ijcaonline.org/research/volume137/number5/i.m.selim-2016-ijca-908387.pdf)
- Khalifa, Taha, Hassanien and Selim, [Deep Galaxy: Classification of Galaxies based on Deep Convolutional Neural Networks](https://arxiv.org/abs/1709.02245)
- Leung and Bovy, [Galaxy10](https://github.com/henrysky/Galaxy10)
- Baillard et al., [The EFIGI catalogue of 4458 nearby galaxies with detailed morphology](https://www.aanda.org/articles/aa/pdf/2011/08/aa16423-10.pdf)
- Bonifacio et al., [Chemical imaging of articular cartilage sections with Raman mapping, employing uni- and multi-variate methods for data analysis](https://doi.org/10.1039/c0an00459f)
- Rabanser, Shchur and Günnemann, [Introduction to Tensor Decompositions and their Applications in Machine Learning](https://arxiv.org/abs/1711.10781)
- Understanding Tensors and Tensor Decomposition, [Part 1](https://iksinc.online/2018/02/12/understanding-tensors-and-tensor-decompositions-part-1/), [Part 2](https://iksinc.online/2018/04/03/understanding-tensors-and-tensor-decompositions-part-2/), [Part 3](https://iksinc.online/2018/05/02/understanding-tensors-and-tensor-decompositions-part-3/)
