import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sparse
import scMODAL.scmodal as scmodal
print(scmodal.__version__)
print(dir(scmodal.model))
import warnings
warnings.filterwarnings("ignore")
import psutil
import gc

def print_memory_usage(prefix=""):
    """Print current CPU memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1e9  # Convert to GB
    percent = process.memory_percent()
    print(f"{prefix}CPU Memory: {mem_gb:.2f} GB ({percent:.1f}% of system RAM)")
    return mem_gb

print_memory_usage("Initial ")

adata_RNA = sc.read_h5ad('/data1/cs690_env/multi.h5ad')
print_memory_usage("After loading RNA h5ad ")

# ✅ Ensure adata_RNA.X stays sparse to prevent densification later
if not sparse.issparse(adata_RNA.X):
    adata_RNA.X = sparse.csr_matrix(adata_RNA.X)

adata_RNA.var.index = adata_RNA.var['_index']

# ✅ Keep sparse matrix — do NOT convert to dense (saves huge RAM)
if hasattr(adata_RNA, "raw") and adata_RNA.raw is not None:
    if sparse.issparse(adata_RNA.raw.X):
        print("✅ Keeping adata_RNA.raw.X as sparse (saves huge memory).")
        adata_RNA.X = adata_RNA.raw.X  # just reference it
    else:
        print("⚠️ adata_RNA.raw.X is dense; converting to sparse CSR matrix.")
        adata_RNA.X = sparse.csr_matrix(adata_RNA.raw.X)
else:
    print("⚠️ No raw layer found; using adata_RNA.X as is.")
    if not sparse.issparse(adata_RNA.X):
        adata_RNA.X = sparse.csr_matrix(adata_RNA.X)

counts_ADT = pd.read_csv('/data1/cs690_env/ADT.csv').T
adata_ADT = ad.AnnData(X = counts_ADT.values)
adata_ADT.obs.index = counts_ADT.index
adata_ADT.var.index = counts_ADT.columns
adata_ADT.obs = adata_RNA.obs.loc[adata_ADT.obs.index]
adata_RNA = adata_RNA[adata_RNA.obs.donor == 'P1']
adata_ADT = adata_ADT[adata_RNA.obs.index]
adata_RNA = adata_RNA[adata_RNA.obs['celltype.l2'].values != 'Doublet']
adata_ADT = adata_ADT[adata_ADT.obs['celltype.l2'].values != 'Doublet']

correspondence = pd.read_csv('/data1/cs690_env/protein_gene_conversion.csv')
correspondence['Protein name'] = correspondence['Protein name'].replace(to_replace={'CD11a-CD18': 'CD11a/CD18', 'CD66a-c-e': 'CD66a/c/e'})
print(correspondence)

rna_protein_correspondence = []

for i in range(correspondence.shape[0]):
    curr_protein_name, curr_rna_names = correspondence.iloc[i]
    if curr_protein_name not in adata_ADT.var_names:
        continue
    if curr_rna_names.find('Ignore') != -1: # some correspondence ignored eg. protein isoform to one gene
        continue
    curr_rna_names = curr_rna_names.split('/') # eg. one protein to multiple genes
    for r in curr_rna_names:
        if r in adata_RNA.var_names:
            rna_protein_correspondence.append([r, curr_protein_name])

rna_protein_correspondence = np.array(rna_protein_correspondence)
print_memory_usage("After correspondence ")

RNA_shared = adata_RNA[:, rna_protein_correspondence[:, 0]].copy()
ADT_shared = adata_ADT[:, rna_protein_correspondence[:, 1]].copy()
RNA_shared.var['feature_name'] = RNA_shared.var.index.values
ADT_shared.var['feature_name'] = ADT_shared.var.index.values
RNA_shared.var_names_make_unique()
ADT_shared.var_names_make_unique()
print_memory_usage("After creating shared ")

RNA_unshared = adata_RNA[:, sorted(set(adata_RNA.var.index) - set(rna_protein_correspondence[:, 0]))].copy()
ADT_unshared = adata_ADT[:, sorted(set(adata_ADT.var.index) - set(rna_protein_correspondence[:, 1]))].copy()
print_memory_usage("After creating unshared ")
# Convert NaNs/Infs to zeros (works for both dense and sparse)
if sparse.issparse(RNA_unshared.X):
    RNA_unshared.X.data = np.nan_to_num(RNA_unshared.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    # Clip only the nonzero entries (sparse-safe)
    upper = np.percentile(RNA_unshared.X.data, 99.9)
    RNA_unshared.X.data = np.clip(RNA_unshared.X.data, a_min=1e-10, a_max=upper)
else:
    RNA_unshared.X = np.nan_to_num(RNA_unshared.X, nan=0.0, posinf=0.0, neginf=0.0)
    RNA_unshared.X = np.clip(RNA_unshared.X, a_min=1e-10, a_max=np.percentile(RNA_unshared.X, 99.9))

# Temporarily patch pandas.cut to drop duplicates
_old_cut = pd.cut
def safe_cut(x, bins, **kwargs):
    try:
        return _old_cut(x, bins, **kwargs)
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            bins = np.unique(bins)
            return _old_cut(x, bins, **kwargs)
        else:
            raise
pd.cut = safe_cut

# Now run without the unsupported 'duplicates' arg
sc.pp.highly_variable_genes(RNA_unshared, flavor='cell_ranger', n_top_genes=3000)

# Restore original pandas.cut
pd.cut = _old_cut

RNA_unshared = RNA_unshared[:, RNA_unshared.var.highly_variable].copy()

RNA_unshared.var['feature_name'] = RNA_unshared.var.index.values
ADT_unshared.var['feature_name'] = ADT_unshared.var.index.values

RNA_counts = RNA_shared.X.sum(axis=1)
ADT_counts = ADT_shared.X.sum(axis=1)
target_sum = np.maximum(np.median(RNA_counts.copy()), 20)

sc.pp.normalize_total(RNA_shared, target_sum=target_sum)
sc.pp.log1p(RNA_shared)

sc.pp.normalize_total(ADT_shared, target_sum=target_sum)
sc.pp.log1p(ADT_shared)

sc.pp.normalize_total(RNA_unshared)
sc.pp.log1p(RNA_unshared)

sc.pp.normalize_total(ADT_unshared)
sc.pp.log1p(ADT_unshared)

adata1 = ad.concat([RNA_shared, RNA_unshared], axis=1)
adata2 = ad.concat([ADT_shared, ADT_unshared], axis=1)
print_memory_usage("After concatenation ")

sc.pp.scale(adata1, max_value=10)
sc.pp.scale(adata2, max_value=10)
print_memory_usage("After scaling ")

# Store shared_gene_num before cleanup
shared_gene_num = RNA_shared.shape[1]

# Delete intermediate objects before training
del RNA_shared, RNA_unshared, ADT_shared, ADT_unshared, counts_ADT
gc.collect()
print_memory_usage("After cleanup before training ")
