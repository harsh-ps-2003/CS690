import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scMODAL.scmodal as scmodal
print(scmodal.__version__)
print(dir(scmodal.model))
import warnings
warnings.filterwarnings("ignore")

adata_RNA = sc.read_h5ad('/data1/cs690_env/multi.h5ad')
adata_RNA.var.index = adata_RNA.var['_index']
adata_RNA.X = adata_RNA.raw.X.toarray()

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

RNA_shared = adata_RNA[:, rna_protein_correspondence[:, 0]].copy()
ADT_shared = adata_ADT[:, rna_protein_correspondence[:, 1]].copy()
RNA_shared.var['feature_name'] = RNA_shared.var.index.values
ADT_shared.var['feature_name'] = ADT_shared.var.index.values
RNA_shared.var_names_make_unique()
ADT_shared.var_names_make_unique()

RNA_unshared = adata_RNA[:, sorted(set(adata_RNA.var.index) - set(rna_protein_correspondence[:, 0]))].copy()
ADT_unshared = adata_ADT[:, sorted(set(adata_ADT.var.index) - set(rna_protein_correspondence[:, 1]))].copy()

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

sc.pp.scale(adata1, max_value=10)
sc.pp.scale(adata2, max_value=10)

model = scmodal.model.Model(model_path="./CITE-seq_PBMC")

model.preprocess(adata1, adata2, shared_gene_num=RNA_shared.shape[1])
model.train()
model.eval()

adata_integrated = ad.AnnData(X=model.latent)
adata_integrated.obs = pd.concat([adata_RNA.obs, adata_ADT.obs])
adata_integrated.obs['modality'] = ['RNA'] * adata_RNA.shape[0] + ['ADT'] * adata_ADT.shape[0]

scmodal.utils.compute_umap(adata_integrated)

sc.pl.umap(adata_integrated, color=['modality', 'celltype.l2'])

from scipy.spatial.distance import cdist

dist_mtx = cdist(model.latent[adata1.shape[0]:, :],
                 model.latent[:adata1.shape[0], :],
                 metric='euclidean') # Transfer labels from RNA to ADT

matching = dist_mtx.argsort()[:, :1]

df1_labels = adata_RNA.obs["celltype.l1"].values
df2_labels = adata_ADT.obs["celltype.l1"].values

print("Label transfer accuracy: ", np.sum(df1_labels == df2_labels[matching.reshape(-1)]) / adata_RNA.shape[0])

# get imputed features in DataFrame
model.get_imputed_df()
imputed_df = model.imputed_df_AtoB

# get ground truth for comparison
true_df = pd.DataFrame(adata2.X[:, :RNA_shared.shape[1]], index=adata2.obs.index, columns=adata2.var.feature_name[:RNA_shared.shape[1]])
true_df = true_df.groupby(true_df.columns, axis=1).mean()

adata_imputed = ad.AnnData(X=np.concatenate([imputed_df.values, true_df.values], axis=1))
adata_imputed.obs = adata_RNA.obs
adata_imputed.var.index = list(imputed_df.columns + '_imputed') + list(true_df.columns)
adata_imputed.obsm['X_umap'] = adata_integrated.obsm['X_umap'][:adata_RNA.shape[0]]

# plot imputed protein abundance levels vs ground truth
sc.pl.umap(adata_imputed, color=['CD102_imputed', 'CD102'])
