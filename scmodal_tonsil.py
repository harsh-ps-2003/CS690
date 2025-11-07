import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sparse
import commot as ct
import os
import gc
import psutil
import multiprocessing as mp
import signal
import sys
import traceback

# ✅ CRITICAL FIX: Limit threads to prevent hitting 1024 thread limit (per-process kernel limit)
# PyTorch/NumPy/OpenBLAS/Annoy create many threads; without limits, can exceed 1024 and get SIGKILL
# Setting all to 1 ensures each library reuses the same single thread, keeping total threads ~70
os.environ["OMP_NUM_THREADS"] = "1"       # OpenMP / MKL threads (Annoy uses this!)
os.environ["MKL_NUM_THREADS"] = "1"       # MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"   # NumExpr threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS threads
os.environ["OPENBLAS_MAIN_FREE"] = "1"    # Keep OpenBLAS worker thread alive (prevents respawn leak)

import torch
torch.set_num_threads(1)                  # PyTorch intra-op parallelism
torch.set_num_interop_threads(1)          # PyTorch inter-op parallelism

# Set multiprocessing to reuse threads (fork) instead of spawning new processes
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    # Already set, ignore
    pass

print("🔒 Thread caps set: OMP/MKL/NumExpr/OpenBLAS/torch = 1 | torch inter-op = 1")
print("   OPENBLAS_MAIN_FREE=1 prevents OpenBLAS from respawning worker threads")
print("   This prevents hitting the 1024 thread limit that causes SIGKILL")
print("   Total threads should stay ~70-100 (well below 1024 limit)")
print("")

# ✅ Signal handler to catch SIGKILL/SIGTERM and log diagnostics
def signal_handler(signum, frame):
    """Catch kill signals and print diagnostics before exit"""
    print("\n" + "="*70)
    print(f"⚠️  RECEIVED SIGNAL {signum} ({signal.Signals(signum).name})")
    print("="*70)
    
    # Print memory stats
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory at kill time:")
        print(f"  RSS: {mem_info.rss / 1e9:.3f}GB")
        print(f"  VMS: {mem_info.vms / 1e9:.3f}GB")
        print(f"  Threads: {process.num_threads()}")
        print(f"  File descriptors: {process.num_fds()}")
        
        vm = psutil.virtual_memory()
        print(f"  System available: {vm.available / 1e9:.1f}GB")
    except Exception as e:
        print(f"  Could not get memory stats: {e}")
    
    # Print stack trace
    print("\nStack trace at kill time:")
    traceback.print_stack(frame)
    
    print("="*70)
    sys.exit(128 + signum)

# Register signal handlers (SIGKILL cannot be caught, but SIGTERM can)
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
# Note: SIGKILL (9) cannot be caught, but we'll see if SIGTERM (15) is used
print("✅ Signal handlers registered (SIGTERM, SIGINT)")
print("")

import scMODAL.scmodal as scmodal
print(scmodal.__version__)

import warnings
warnings.filterwarnings("ignore")

def print_memory_usage(prefix=""):
    """Print current CPU memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1e9  # GB
    percent = process.memory_percent()
    virtual = psutil.virtual_memory()
    system_available = virtual.available / 1e9
    print(f"{prefix}CPU Memory: {mem_gb:.2f} GB ({percent:.1f}%), System Available: {system_available:.1f}GB")
    return mem_gb

print_memory_usage("Initial ")

adata_CODEX = ad.read_h5ad('/data1/cs690_env/adata_codex.h5ad')
print_memory_usage("After loading CODEX ")

adata_RNA = ad.read_h5ad('/data1/cs690_env/adata_rna.h5ad')
print_memory_usage("After loading RNA ")

adata_ATAC = ad.read_h5ad('/data1/cs690_env/adata_atac.h5ad')
print_memory_usage("After loading ATAC ")

correspondence = pd.read_csv('/data1/cs690_env/protein_gene_conversion.csv', )
correspondence['Protein name'] = correspondence['Protein name'].replace(to_replace={'CD11a-CD18': 'CD11a/CD18', 'CD66a-c-e': 'CD66a/c/e'})
rna_protein_correspondence = []

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]
    if curr_protein_name not in adata_CODEX.var_names:
        continue
    if curr_rna_names.find('Ignore') != -1: # some correspondence ignored eg. protein isoform to one gene
        continue
    curr_rna_names = curr_rna_names.split('/') # eg. one protein to multiple genes
    for r in curr_rna_names:
        if r in adata_RNA.var_names:
            rna_protein_correspondence.append([r, curr_protein_name])

rna_protein_correspondence = np.array(rna_protein_correspondence)

# ✅ Use views where possible to reduce VMS (avoid unnecessary copies)
# Only copy if we need to modify the data structure
adata_RNA_shared = adata_RNA[:, rna_protein_correspondence[:, 0]].copy()
adata_CODEX_shared = adata_CODEX[:, rna_protein_correspondence[:, 1]].copy()
# Force garbage collection after copy to reduce VMS fragmentation
gc.collect()

adata_RNA_shared.var_names_make_unique()
adata_CODEX_shared.var_names_make_unique()
df_cellchat = ct.pp.ligand_receptor_database(species='human', signaling_type='Secreted Signaling', database='CellPhoneDB_v4.0')
print(df_cellchat.shape)

# Preserve more genes in ligand-receptor database in scRNA-seq data for cell-cell communication inference

# ✅ Reduce VMS: Use views where possible, only copy when necessary
adata_RNA_unshared = adata_RNA[:, sorted(set(adata_RNA.var.index) - set(rna_protein_correspondence[:, 0]))].copy()
gc.collect()  # Free memory after copy
adata_RNA_lr = adata_RNA_unshared[:, adata_RNA_unshared.var.index.isin(np.unique(df_cellchat['0'].values)) | adata_RNA_unshared.var.index.isin(np.unique(df_cellchat['1'].values))].copy()
gc.collect()  # Free memory after copy

# ✅ CRITICAL FIX: Use sparse-safe sum instead of .toarray() to avoid massive memory consumption
if sparse.issparse(adata_RNA_lr.X):
    # Sparse matrix: sum along axis 0 without converting to dense
    gene_sums = np.array(adata_RNA_lr.X.sum(axis=0)).flatten()
else:
    gene_sums = np.sum(adata_RNA_lr.X, axis=0)

adata_RNA_lr_variable = adata_RNA_lr[:, (gene_sums > 10)].var.index
del gene_sums  # Cleanup
print_memory_usage("After LR filtering ")

sc.pp.highly_variable_genes(adata_RNA_unshared, flavor='seurat_v3', n_top_genes=1000)
adata_RNA_unshared = adata_RNA_unshared[:, adata_RNA_unshared.var.highly_variable | adata_RNA_unshared.var.index.isin(adata_RNA_lr_variable)].copy()
gc.collect()  # Free memory after copy to reduce VMS
# Finding ATAC / CODEX shared features is helpful for normalizing ATAC data

atac_protein_correspondence = []

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]
    if curr_protein_name not in adata_CODEX.var_names:
        continue
    if curr_rna_names.find('Ignore') != -1: # some correspondence ignored eg. protein isoform to one gene
        continue
    curr_rna_names = curr_rna_names.split('/') # eg. one protein to multiple genes
    for r in curr_rna_names:
        if r in adata_ATAC.var_names:
            atac_protein_correspondence.append([r, curr_protein_name])

atac_protein_correspondence = np.array(atac_protein_correspondence)
adata_ATAC_shared = adata_ATAC[:, atac_protein_correspondence[:, 0]].copy()
adata_CODEX_ATAC_shared = adata_CODEX[:, atac_protein_correspondence[:, 1]].copy()
gc.collect()  # Free memory after copy

adata_ATAC_shared.var_names_make_unique()
adata_CODEX_ATAC_shared.var_names_make_unique()
adata_ATAC_unshared = adata_ATAC[:, sorted(set(adata_ATAC.var.index) - set(atac_protein_correspondence[:, 0]))].copy()
gc.collect()  # Free memory after copy
sc.pp.highly_variable_genes(adata_ATAC_unshared, flavor='seurat_v3', n_top_genes=1000)
adata_ATAC_unshared = adata_ATAC_unshared[:, adata_ATAC_unshared.var.highly_variable].copy()
gc.collect()  # Free memory after copy to reduce VMS

# ✅ CRITICAL FIX: Compute target_sum without creating huge dense float64 arrays (reduces VMS)
# Use sparse-safe computation with float32 to avoid 6-7GB temporary allocations
if sparse.issparse(adata_CODEX_shared.X):
    # Compute exp-1 and sum per row without converting entire matrix to dense
    # This avoids creating a 6-7GB float64 dense matrix
    exp_data = np.exp(adata_CODEX_shared.X.data.astype(np.float32)) - 1.0
    row_sums = np.add.reduceat(exp_data, adata_CODEX_shared.X.indptr[:-1])
    target_sum_rna = float(np.median(row_sums))
    del exp_data, row_sums  # Free immediately
    gc.collect()
else:
    # Dense case: use float32 to reduce memory
    target_sum_rna = float(np.median((np.exp(adata_CODEX_shared.X.astype(np.float32)) - 1.0).sum(axis=1)))

sc.pp.normalize_total(adata_RNA_shared, target_sum=target_sum_rna)
sc.pp.log1p(adata_RNA_shared)

sc.pp.normalize_total(adata_RNA_unshared)
sc.pp.log1p(adata_RNA_unshared)

adata_RNA = ad.concat([adata_RNA_shared, adata_RNA_unshared], axis=1)
adata_RNA.obs["celltype"] = adata_RNA_shared.obs["celltype"]
gc.collect()  # Free memory after concat to reduce VMS

# ✅ Save shared feature count before cleanup (needed for model training)
n_rna_codex_shared = adata_RNA_shared.shape[1]
print(f"Number of RNA-CODEX shared features: {n_rna_codex_shared}")

adata_CODEX = adata_CODEX_shared # CODEX data do not contain unlinked features with RNA data
adata_CODEX.obs["celltype"] = adata_CODEX.obs["celltype"]

print_memory_usage("Before scaling ")

sc.pp.scale(adata_RNA, max_value=10)
sc.pp.scale(adata_CODEX, max_value=10)
print_memory_usage("After scaling RNA/CODEX ")

# ✅ CRITICAL FIX: Compute target_sum without creating huge dense float64 arrays (reduces VMS)
# Use sparse-safe computation with float32 to avoid 6-7GB temporary allocations
if sparse.issparse(adata_CODEX_ATAC_shared.X):
    # Compute exp-1 and sum per row without converting entire matrix to dense
    # This avoids creating a 6-7GB float64 dense matrix
    exp_data = np.exp(adata_CODEX_ATAC_shared.X.data.astype(np.float32)) - 1.0
    row_sums = np.add.reduceat(exp_data, adata_CODEX_ATAC_shared.X.indptr[:-1])
    target_sum_atac = float(np.median(row_sums))
    del exp_data, row_sums  # Free immediately
    gc.collect()
else:
    # Dense case: use float32 to reduce memory
    target_sum_atac = float(np.median((np.exp(adata_CODEX_ATAC_shared.X.astype(np.float32)) - 1.0).sum(axis=1)))

sc.pp.normalize_total(adata_ATAC_shared, target_sum=target_sum_atac)
sc.pp.log1p(adata_ATAC_shared)

sc.pp.normalize_total(adata_ATAC_unshared)
sc.pp.log1p(adata_ATAC_unshared)

adata_ATAC = ad.concat([adata_ATAC_shared, adata_ATAC_unshared], axis=1)
adata_ATAC.obs["dataset"] = "ATAC"
adata_ATAC.obs["celltype"] = adata_ATAC_shared.obs["celltype"]
gc.collect()  # Free memory after concat to reduce VMS

sc.pp.scale(adata_ATAC, max_value=10)

# Use set intersection for shared features
RNA_ATAC_shared = sorted(list(set(adata_RNA.var.index) & set(adata_ATAC.var.index)))
print(adata_RNA.shape, adata_ATAC.shape, len(RNA_ATAC_shared))

adata_CODEX.obs['modality'] = 'CODEX'
adata_RNA.obs['modality'] = 'RNA'
adata_ATAC.obs['modality'] = 'ATAC'
adata_RNA_ATAC_shared = ad.concat([adata_RNA[:, RNA_ATAC_shared], adata_ATAC[:, RNA_ATAC_shared]])
gc.collect()  # Free memory after concat
sc.tl.pca(adata_RNA_ATAC_shared, n_comps=30)
gc.collect()  # Free memory after PCA to reduce VMS
print_memory_usage("After PCA ")

# ✅ CRITICAL: Cleanup intermediate objects before training to free RAM
print("\n" + "="*70)
print("CLEANING UP INTERMEDIATE OBJECTS BEFORE TRAINING")
print("="*70)
del adata_RNA_shared, adata_RNA_unshared, adata_RNA_lr, adata_RNA_lr_variable
del adata_CODEX_shared, adata_ATAC_shared, adata_ATAC_unshared
del adata_CODEX_ATAC_shared, atac_protein_correspondence
del rna_protein_correspondence, correspondence, df_cellchat
gc.collect()
print_memory_usage("After cleanup before training ")
print("="*70 + "\n")

# ✅ CRITICAL FIX: Convert paired_input_MNN to dense ONCE before training to avoid repeated conversions
print("Preparing paired_input_MNN matrices (converting sparse to dense if needed)...")
codex_rna_shared = adata_CODEX.X[:, :n_rna_codex_shared]
rna_shared = adata_RNA.X[:, :n_rna_codex_shared]

# ✅ CRITICAL FIX: Convert to float32 dense arrays (half the size of float64, reduces VMS)
if sparse.issparse(codex_rna_shared):
    print(f"  Converting CODEX-RNA shared features to dense float32: {codex_rna_shared.shape} (sparse) -> dense")
    codex_rna_shared = codex_rna_shared.astype(np.float32).toarray()
    gc.collect()  # Free sparse matrix memory immediately
if sparse.issparse(rna_shared):
    print(f"  Converting RNA shared features to dense float32: {rna_shared.shape} (sparse) -> dense")
    rna_shared = rna_shared.astype(np.float32).toarray()
    gc.collect()  # Free sparse matrix memory immediately

# ✅ CRITICAL: Copy PCA data and delete large object to reduce VMS
pca_rna = adata_RNA_ATAC_shared.obsm['X_pca'][:adata_RNA.shape[0]].copy()
pca_atac = adata_RNA_ATAC_shared.obsm['X_pca'][adata_RNA.shape[0]:].copy()

# ✅ CRITICAL: Delete the large concatenated object to reduce VMS
del adata_RNA_ATAC_shared
gc.collect()

print_memory_usage("After preparing MNN pairs ")

# ✅ CRITICAL FIX: Reduce batch_size to fit in memory limit
# batch_size=128 uses ~1.5GB, batch_size=64 uses ~1.2GB
# If hitting 2GB cgroup/session limit, use batch_size=64
model = scmodal.model.Model(batch_size=64, training_steps=10000, lambdaMNN=5, lambdaGAN=0.5, model_path="./tonsil_tutorial")
print(f"Model initialized with batch_size=64 (reduced to fit in 2GB memory limit)")

model.integrate_datasets_feats(input_feats=[adata_CODEX.X, adata_RNA.X, adata_ATAC.X],
                              paired_input_MNN=[[codex_rna_shared, rna_shared],
                                                [pca_rna, pca_atac]], )

adata_integrated = ad.AnnData(X=model.latent)
adata_integrated.obs['modality'] = ['CODEX'] * adata_CODEX.shape[0] + ['RNA'] * adata_RNA.shape[0] + ['ATAC'] * adata_ATAC.shape[0]
adata_integrated.obs['celltype'] = list(adata_CODEX.obs['celltype'].values) + list(adata_RNA.obs['celltype'].values) + list(adata_ATAC.obs['celltype'].values)

scmodal.utils.compute_umap(adata_integrated)

sc.pl.umap(adata_integrated, color=['modality', "celltype"])

import numpy as np
import gc
import torch

from sklearn.neighbors import NearestNeighbors

# Function to free memory, call frequently
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# For batch RAM clearing, call after every batch
n_code_cells = adata_CODEX.shape[0]
n_rna_cells = adata_RNA.shape[0]

top_k = 1  # or more, as needed

# Build a KNN model for RNA cells using sklearn (fast, low RAM)
knn = NearestNeighbors(n_neighbors=top_k, algorithm='auto', metric='euclidean')
knn.fit(model.latent[n_code_cells:(n_code_cells + n_rna_cells), :])
# Find k nearest RNA neighbors for all CODEX cells (batched internally by sklearn)
distances, indices = knn.kneighbors(model.latent[:n_code_cells, :])

# indices.shape = (n_code_cells, top_k)
# Fetch RNA celltype labels
celltype_array = adata_RNA.obs["celltype"].values.astype(str)[indices]

# Mode function for categorical in numpy
def string_mode(x):
    labels, counts = np.unique(x, return_counts=True)
    return labels[np.argmax(counts)]

transfered = np.apply_along_axis(string_mode, 1, celltype_array)

clear_memory()  # Encourage garbage collection

# --- Visualization (unchanged) ---
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import matplotlib.pyplot as plt

colours = ListedColormap(['#393b79', '#ff7f0e', '#98df8a', '#8c564b', '#e7cb94', "tab:purple"])
le = preprocessing.LabelEncoder()
le.fit(sorted(set(transfered)))
label = le.transform(transfered)

f = plt.figure(figsize=(10,10))
ax1 = f.add_subplot(1,1,1)
scatter1 = ax1.scatter(adata_CODEX.obsm['spatial'][:, 0], adata_CODEX.obsm['spatial'][:, 1], c=label, cmap=colours, label=transfered, s=1.5, rasterized=True)
ax1.tick_params(axis='both',bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False, grid_alpha=0)
l1 = f.legend(handles=scatter1.legend_elements()[0], labels=sorted(set(transfered)), loc="upper left", bbox_to_anchor=(0.9, 0.45),
                    markerscale=3., title_fontsize=30, fontsize=30, frameon=False, ncol=1)
l1._legend_box.align = "left"
ax1.set_title("Transfered annotation", fontsize=45)
f.tight_layout()
ax1.axis('off')
plt.show()

clear_memory()  # Final cache clear after plotting
