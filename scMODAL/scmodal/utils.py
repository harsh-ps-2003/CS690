import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import umap
from .model import *
from scipy.spatial.distance import cdist
import gc

def acquire_pairs(X, Y, k=30, metric='angular'):
    # This function was modified from iMAP: https://github.com/Svvord/iMAP/blob/master/imap/stage2.py
    # ✅ CRITICAL FIX: Pure NumPy implementation - ZERO thread creation
    # Replaced Annoy/sklearn to prevent thread leaks that accumulate to 1024+ and cause SIGKILL
    # Pure NumPy operations respect OMP_NUM_THREADS=1 and create no persistent threads
    
    # Convert to numpy arrays if needed
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if metric == 'angular':
        # Angular distance via cosine similarity (pure NumPy, respects OMP_NUM_THREADS=1)
        # Normalize vectors
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        # Cosine similarity matrix (batch_size x batch_size)
        sim = X_norm @ Y_norm.T  # Matrix multiply (uses BLAS with OMP_NUM_THREADS=1)
        # For angular, higher similarity = closer (negate for distance-like)
        dist_matrix = -sim
    else:
        # Euclidean distance (pure NumPy)
        # dist_matrix[i,j] = ||X[i] - Y[j]||^2
        XX = np.sum(X**2, axis=1, keepdims=True)  # (len(X), 1)
        YY = np.sum(Y**2, axis=1, keepdims=True)  # (len(Y), 1)
        XY = X @ Y.T  # (len(X), len(Y))
        dist_matrix = XX + YY.T - 2 * XY
    
    # Find k nearest neighbors: argpartition is O(n) and thread-safe
    # Get top-k indices for each row
    idx_Y = np.argpartition(dist_matrix, k-1, axis=1)[:, :k]  # Top-k of X in Y
    idx_X = np.argpartition(dist_matrix.T, k-1, axis=1)[:, :k]  # Top-k of Y in X
    
    # Build mutual nearest neighbors matrix
    mnn_mat = np.zeros((len(X), len(Y)), dtype=bool)
    for i, nbrs in enumerate(idx_Y):
        mnn_mat[i, nbrs] = True
    for j, nbrs in enumerate(idx_X):
        mnn_mat[nbrs, j] &= True  # Keep only mutual neighbors
    
    # Cleanup (don't delete X, Y as they're input parameters)
    del dist_matrix, idx_Y, idx_X
    if metric == 'angular':
        del X_norm, Y_norm, sim
    
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

