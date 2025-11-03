import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import umap
from .model import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import gc

def acquire_pairs(X, Y, k=30, metric='angular'):
    # This function was modified from iMAP: https://github.com/Svvord/iMAP/blob/master/imap/stage2.py
    # ✅ CRITICAL FIX: Replaced Annoy with sklearn to prevent thread leaks
    # Annoy's build() spawns worker threads that never die, accumulating to 1024+ and causing SIGKILL
    # sklearn's NearestNeighbors doesn't spawn persistent threads, solving the issue
    
    # Convert 'angular' metric to sklearn equivalent ('cosine' for angular distance)
    sklearn_metric = 'cosine' if metric == 'angular' else metric
    
    # Find k nearest neighbors of X in Y
    nn1 = NearestNeighbors(n_neighbors=k, metric=sklearn_metric, n_jobs=1).fit(Y)
    idx_Y = nn1.kneighbors(X, return_distance=False)
    
    # Find k nearest neighbors of Y in X
    nn2 = NearestNeighbors(n_neighbors=k, metric=sklearn_metric, n_jobs=1).fit(X)
    idx_X = nn2.kneighbors(Y, return_distance=False)
    
    # Build mutual nearest neighbors matrix
    mnn_mat = np.zeros((len(X), len(Y)), dtype=bool)
    for i, nbrs in enumerate(idx_Y):
        mnn_mat[i, nbrs] = True
    for j, nbrs in enumerate(idx_X):
        mnn_mat[nbrs, j] &= True  # Keep only mutual neighbors
    
    # Cleanup
    del nn1, nn2, idx_Y, idx_X
    
    return mnn_mat.astype(int)
     
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels

def compute_umap(adata, rep=None):
    import umap

    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    if rep is None:
        X_umap = reducer.fit_transform(adata.X)
    else:
        X_umap = reducer.fit_transform(adata.obsm[rep])

    adata.obsm['X_umap'] = X_umap

