import os
import time
import gc
import numpy as np
import scipy.sparse as sparse
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .networks import *
from .utils import *

# ✅ CRITICAL FIX: malloc_trim to return freed memory to OS and reduce VMS
# glibc's malloc keeps freed memory in arenas, never returning it to OS
# This causes VMS to stay high even when RSS is low, triggering OOM kills
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    def malloc_trim():
        """Return freed memory to OS to reduce VMS"""
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass  # Silently fail if not available
except Exception:
    # Fallback if ctypes/CDLL not available (e.g., macOS)
    def malloc_trim():
        pass

# Add CPU memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Install with: pip install psutil")


def thread_count():
    """Return number of live threads in this process."""
    try:
        return len(os.listdir("/proc/self/task"))
    except (FileNotFoundError, OSError):  # macOS / non-proc systems
        import threading
        return threading.active_count()


def get_cpu_memory_stats():
    """Get current CPU memory statistics"""
    if not PSUTIL_AVAILABLE:
        return None
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1e9  # GB
    percent = process.memory_percent()
    
    # Get system memory
    virtual = psutil.virtual_memory()
    system_total = virtual.total / 1e9
    system_available = virtual.available / 1e9
    
    # Get detailed memory info
    mem_full = process.memory_full_info() if hasattr(process, 'memory_full_info') else None
    
    return {
        'process_memory_gb': mem_gb,
        'process_percent': percent,
        'system_total_gb': system_total,
        'system_available_gb': system_available,
        'system_used_percent': virtual.percent,
        'rss_gb': mem_info.rss / 1e9,
        'vms_gb': mem_info.vms / 1e9,
        'shared_gb': mem_info.shared / 1e9 if hasattr(mem_info, 'shared') else 0,
        'num_fds': process.num_fds() if hasattr(process, 'num_fds') else None,
        'num_threads': process.num_threads() if hasattr(process, 'num_threads') else None,
    }


def get_gpu_memory_stats(device=None):
    """Get current GPU memory statistics"""
    if not torch.cuda.is_available():
        return None
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
    reserved = torch.cuda.memory_reserved(device) / 1e9  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9  # GB (peak)
    max_reserved = torch.cuda.max_memory_reserved(device) / 1e9  # GB (peak)
    
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
    free_memory = total_memory - reserved
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free_memory,
        'total': total_memory,
        'max_allocated': max_allocated,
        'max_reserved': max_reserved,
        'usage_percent': (reserved / total_memory) * 100
    }


def print_cpu_memory(prefix=""):
    """Print CPU memory statistics"""
    stats = get_cpu_memory_stats()
    if stats is None:
        return
    
    msg = f"{prefix}CPU Memory: Process={stats['process_memory_gb']:.2f}GB ({stats['process_percent']:.1f}%), "
    msg += f"System={stats['system_used_percent']:.1f}% used, Available={stats['system_available_gb']:.1f}GB"
    print(msg)
    return stats


def print_gpu_memory(prefix="", device=None, print_peak=False):
    """Print GPU memory statistics"""
    stats = get_gpu_memory_stats(device)
    if stats is None:
        return
    
    msg = f"{prefix}GPU Memory: Allocated={stats['allocated']:.2f}GB, Reserved={stats['reserved']:.2f}GB, "
    msg += f"Free={stats['free']:.2f}GB/{stats['total']:.2f}GB ({stats['usage_percent']:.1f}%)"
    
    if print_peak:
        msg += f" | Peak: Allocated={stats['max_allocated']:.2f}GB, Reserved={stats['max_reserved']:.2f}GB"
    
    print(msg)
    return stats


def reset_peak_memory_stats(device=None):
    """Reset peak memory statistics"""
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)

class Model(object):
    def __init__(self, batch_size=500, training_steps=10000, seed=1234, n_latent=20,
                 lambdaAE = 10.0, lambdaLA = 10.0, lambdaMNN = 1.0, lambdaGeo = 10.0, lambdaGAN = 1.0, n_KNN = 30,
                 model_path="models", data_path="data", result_path="results"):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaAE = lambdaAE
        self.lambdaLA = lambdaLA
        self.lambdaMNN = lambdaMNN
        self.lambdaGeo = lambdaGeo
        self.lambdaGAN = lambdaGAN
        self.n_KNN = n_KNN
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path


    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num
                   ):
        self.adata_A = adata_A_input.copy()
        self.adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        # ✅ Convert to float32 to reduce VMS (half the size of float64)
        if sparse.issparse(self.adata_A.X):
            self.emb_A = self.adata_A.X.astype(np.float32).toarray()
        else:
            self.emb_A = self.adata_A.X.astype(np.float32)
        if sparse.issparse(self.adata_B.X):
            self.emb_B = self.adata_B.X.astype(np.float32).toarray()
        else:
            self.emb_B = self.adata_B.X.astype(np.float32)

    def preprocess_additional_inputs(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num,
                   layer_adata_A_MNN=None, 
                   layer_adata_B_MNN=None, 
                   ):
        # For ATAC-seq data, an option is to let adata_X_input be LSI matrices, 
        # layer_adata_X_MNN be the layer name storing gene activity matrices
        # The first K=shared_gene_num features in self.feat_A_MNN and self.feat_B_MNN should be positively related .

        assert ((layer_adata_A_MNN is not None) or (layer_adata_B_MNN is not None)), "One of the layer names should be feeded; otherwise, use .preprocess() function."
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        self.emb_A = adata_A.X
        self.emb_B = adata_B.X
        if layer_adata_A_MNN is None:
            self.feat_A_MNN = self.emb_A
        else:
            self.feat_A_MNN = adata_A.obsm[layer_adata_A_MNN]
        if layer_adata_B_MNN is None:
            self.feat_B_MNN = self.emb_B
        else:
            self.feat_B_MNN = adata_B.obsm[layer_adata_B_MNN]


    def train(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        
        # Memory monitoring - initial state
        device = None
        if torch.cuda.is_available():
            device = self.device if isinstance(self.device, int) else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
            print_gpu_memory("Initial ", device=device)
            
            # Get GPU info
            props = torch.cuda.get_device_properties(device)
            print(f"GPU: {props.name} | Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"Batch size: {self.batch_size} | Training steps: {self.training_steps}")
            print("-" * 70)
        
        # Monitor CPU memory (the real killer)
        print_cpu_memory("Initial ")
        print("-" * 70)
        
        self.E_A = encoder(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.D_Z = discriminator(self.n_latent).to(self.device)
        
        # Memory after model initialization
        if torch.cuda.is_available() and device is not None:
            print_gpu_memory("After model init ", device=device)
        
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)
        optimizer_D = optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_Z.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            # Monitor memory at start of step (before forward pass)
            step_start_memory = None
            if torch.cuda.is_available() and (step == 0 or step % 500 == 0):
                step_start_memory = get_gpu_memory_stats(device)
            
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            
            # Memory after data transfer
            if torch.cuda.is_available() and step == 0:
                print_gpu_memory("After data to GPU ", device=device)
            
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)
            
            # Memory-intensive pairwise distance computations
            K_A = torch.mean((x_A.view(self.batch_size, 1, -1) - x_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A = torch.exp(-K_A/2)
            K_B_z = torch.mean((z_B.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B_z = torch.exp(-K_B_z/2)
            K_B = torch.mean((x_B.view(self.batch_size, 1, -1) - x_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B = torch.exp(-K_B/2)
            K_A_z = torch.mean((z_A.view(self.batch_size, 1, -1) - z_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A_z = torch.exp(-K_A_z/2)
            
            # Memory after pairwise computations (critical point)
            if torch.cuda.is_available() and step == 0:
                print_gpu_memory("After pairwise matrices ", device=device)

            # discriminator loss:
            # Use detached tensors to avoid retaining computation graph
            z_A_detached = z_A.detach()
            z_B_detached = z_B.detach()
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = (torch.log(1 + torch.exp(-self.D_Z(z_A_detached))) + torch.log(1 + torch.exp(self.D_Z(z_B_detached)))).mean()
                loss_D.backward()
                optimizer_D.step()
            del z_A_detached, z_B_detached

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            loss_G_GAN = -(torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()

            # geometric structure loss
            loss_Geo = - (torch.clamp(cos(K_A, K_A_z), max=0.975).mean() + torch.clamp(cos(K_B, K_B_z), max=0.975).mean())

            # MNN loss
            #Acquire_pairs creates numpy arrays on CPU - this accumulates in RAM
            Sim_np = acquire_pairs(self.emb_A[index_A, :self.shared_gene_num], self.emb_B[index_B, :self.shared_gene_num], k=self.n_KNN)
            Sim = torch.from_numpy(Sim_np).float().to(self.device)
            del Sim_np  # Immediately delete numpy array to free CPU memory
            z_dist = torch.mean((z_A.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            loss_MNN = torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
            
            # Delete pairwise matrices immediately after loss computation (before backward)
            del K_A, K_B, K_A_z, K_B_z
            
            # Memory before backward pass (peak usage point)
            pre_backward_memory = None
            if torch.cuda.is_available():
                pre_backward_memory = get_gpu_memory_stats(device)
                # Warn if memory usage is high
                if pre_backward_memory['usage_percent'] > 90:
                    print(f"⚠️  WARNING at step {step}: GPU memory usage > 90% ({pre_backward_memory['usage_percent']:.1f}%)")
                    # Emergency cleanup
                    torch.cuda.empty_cache()
                    gc.collect()
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()
            
            # ✅ Log before deleting tensors (so loss variables still exist)
            if not step % 2000:
                print(
                    "step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
                    % (
                        step,
                        float(loss_D),
                        float(loss_G_GAN),
                        float(self.lambdaAE) * float(loss_AE),
                        float(self.lambdaGeo) * float(loss_Geo),
                        float(self.lambdaLA) * float(loss_LA),
                        float(self.lambdaMNN) * float(loss_MNN),
                    )
                )
                if torch.cuda.is_available():
                    print_gpu_memory(f"Step {step} ", device=device, print_peak=True)
                # Monitor CPU memory to catch OOM before it happens
                cpu_stats = print_cpu_memory(f"Step {step} ")
                if cpu_stats and cpu_stats['system_available_gb'] < 2.0:
                    print(f"WARNING: System RAM running low! Available: {cpu_stats['system_available_gb']:.2f}GB")
                    print("    Forcing aggressive garbage collection...")
                    gc.collect()
                    malloc_trim()  # Return memory to OS
                print("-" * 70)
            
            # ✅ Immediately free memory after printing
            del loss_G, loss_AE, loss_AE_A, loss_AE_B, loss_LA, loss_LA_AtoB, loss_LA_BtoA
            del loss_G_GAN, loss_Geo, loss_MNN, Sim, z_dist
            del x_A, x_B, z_A, z_B, x_AtoB, x_BtoA, x_Arecon, x_Brecon, z_AtoB, z_BtoA
            
            # ✅ Memory check for the first iteration
            if torch.cuda.is_available() and step == 0:
                print_gpu_memory("After backward+optimizer ", device=device)
            
            # ✅ Mid-step monitoring to detect gradual GPU AND CPU memory leaks
            if step % 500 == 0 and step > 0:
                if torch.cuda.is_available():
                    current_memory = get_gpu_memory_stats(device)
                    if current_memory:
                        usage_pct = current_memory["usage_percent"]
                        if hasattr(self, "_prev_gpu_memory"):
                            memory_growth = current_memory["reserved"] - self._prev_gpu_memory["reserved"]
                            if memory_growth > 0.1:
                                print(
                                    f"Step {step}: ⚠️ GPU memory growth +{memory_growth:.2f}GB since last check"
                                )
                        self._prev_gpu_memory = current_memory.copy()
                        if usage_pct > 90:
                            print(
                                f"⚠️ High GPU usage ({usage_pct:.1f}%), forcing cache cleanup..."
                            )
                            torch.cuda.empty_cache()
                            gc.collect()
                
                # ⚠️ CRITICAL: Monitor CPU memory growth (the real issue)
                current_cpu_memory = get_cpu_memory_stats()
                if current_cpu_memory:
                    if hasattr(self, "_prev_cpu_memory"):
                        cpu_growth = current_cpu_memory["process_memory_gb"] - self._prev_cpu_memory["process_memory_gb"]
                        if cpu_growth > 0.5:  # If process grew by more than 0.5GB
                            print(
                                f"Step {step}: ⚠️ CPU memory growth +{cpu_growth:.2f}GB since last check"
                            )
                            print(f"    Process using {current_cpu_memory['process_memory_gb']:.2f}GB, forcing GC...")
                            gc.collect()
                    self._prev_cpu_memory = current_cpu_memory.copy()
                    
                    # ✅ NEW: Check VMS growth - this is what triggers the kill!
                    if current_cpu_memory['vms_gb'] > 10.0:  # VMS over 10GB
                        print(
                            f"⚠️ WARNING: VMS is {current_cpu_memory['vms_gb']:.2f}GB - triggering aggressive cleanup!"
                        )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        malloc_trim()  # CRITICAL: Return memory to OS to reduce VMS
                    
                    # Emergency cleanup if system RAM is low
                    if current_cpu_memory['system_available_gb'] < 3.0:
                        print(
                            f"⚠️ CRITICAL: System RAM low ({current_cpu_memory['system_available_gb']:.2f}GB available)!"
                        )
                        print("    Forcing emergency cleanup...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        malloc_trim()  # Return memory to OS
            
            # ✅ Extra cleanup every few steps - INCREASE FREQUENCY
            if step % 10 == 0 and step > 0:
                # NEW: Ultra-frequent malloc_trim to keep VMS low
                malloc_trim()
            if step % 25 == 0 and step > 0:
                # Light cleanup - just CUDA cache + malloc_trim
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                malloc_trim()  # Add here too
            if step % 100 == 0 and step > 0:
                # Aggressive cleanup - both CPU and GPU
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # ✅ CRITICAL: Return freed memory to OS to reduce VMS
                malloc_trim()
        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)
        
        # Final memory statistics
        if torch.cuda.is_available() and device is not None:
            print("-" * 70)
            print_gpu_memory("Final ", device=device, print_peak=True)
            stats = get_gpu_memory_stats(device)
            if stats:
                print(f"Peak memory utilization: {stats['max_reserved']/stats['total']*100:.1f}%")
                print("=" * 70)

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            
            # Move models to CPU before saving to avoid OOM
            self.E_A.cpu()
            self.E_B.cpu()
            self.G_A.cpu()
            self.G_B.cpu()
            torch.cuda.empty_cache()
            
            state = {
                'E_A': self.E_A.state_dict(),
                'E_B': self.E_B.state_dict(),
                'G_A': self.G_A.state_dict(),
                'G_B': self.G_B.state_dict()
            }
            
            torch.save(state, os.path.join(self.model_path, "ckpt.pth"))


    def eval(self):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        self.E_A = encoder(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.E_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_A'])
        self.E_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['E_B'])
        self.G_A.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_A'])
        self.G_B.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['G_B'])

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)

        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)

        end_time = time.time()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

    def get_imputed_df(self, 
                       scale = 'scaled' # if scale=='log', then restore expression after log1p
                       ):

        x_BtoA = self.data_Aspace[self.emb_A.shape[0]:]
        x_AtoB = self.data_Bspace[:self.emb_A.shape[0]]
        if scale == 'log':
            x_BtoA = x_BtoA * self.adata_A.var['std'].values.reshape(1, -1) + self.adata_A.var['mean'].values.reshape(1, -1)
            x_AtoB = x_AtoB * self.adata_B.var['std'].values.reshape(1, -1) + self.adata_B.var['mean'].values.reshape(1, -1)
        imputed_df_BtoA = pd.DataFrame(x_BtoA, index=self.adata_B.obs.index, columns=self.adata_A.var.feature_name)
        imputed_df_BtoA = imputed_df_BtoA.groupby(imputed_df_BtoA.columns, axis=1).mean()
        imputed_df_AtoB = pd.DataFrame(x_AtoB, index=self.adata_A.obs.index, columns=self.adata_B.var.feature_name)
        imputed_df_AtoB = imputed_df_AtoB.groupby(imputed_df_AtoB.columns, axis=1).mean()
        self.imputed_df_BtoA = imputed_df_BtoA
        self.imputed_df_AtoB = imputed_df_AtoB

    def integrate_datasets_links(self, # Use this function for N >= 3 datasets when provided features links for MNN
                                 input_feats,
                                 feat_links_MNN, # A list of index pairs for feature linkages between features in "inputs_MNN"
                                 input_MNN=None, # A list of features matrices for finding MNN pairs between datasets; set as the same as input_feats if "input_MNN=None"
                                 ):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        num_datasets = len(input_feats)
        assert len(feat_links_MNN) == (num_datasets-1)
        self.E_dict = {}
        self.G_dict = {}
        params_G = []
        for i in range(num_datasets):
            self.E_dict[i] = encoder(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.E_dict[i].parameters()
            self.G_dict[i] = generator(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.G_dict[i].parameters()
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)

        self.D_dict = {}
        params_D = []
        for i in range(num_datasets-1):
            self.D_dict[i] = discriminator(self.n_latent).to(self.device)
            params_D += self.D_dict[i].parameters()
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.001)

        for i in range(num_datasets):
            self.E_dict[i].train()
            self.G_dict[i].train()
        for i in range(num_datasets-1):
            self.D_dict[i].train()

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            x_dict = {}
            z_dict = {}
            K_dict = {}
            K_z_dict = {}
            if input_MNN != None:
                assert len(input_MNN) == num_datasets
                x_MNN_dict = {}
            for i in range(num_datasets):
                index_i = np.random.choice(np.arange(input_feats[i].shape[0]), size=self.batch_size)
                
                # ✅ Handle sparse matrices properly - convert to float32 to reduce VMS
                batch_data = input_feats[i][index_i, :]
                if sparse.issparse(batch_data):
                    batch_data = batch_data.astype(np.float32).toarray()
                else:
                    # Ensure float32 even if already dense
                    batch_data = batch_data.astype(np.float32)
                x_dict[i] = torch.from_numpy(batch_data).float().to(self.device)
                
                if input_MNN != None:
                    assert input_MNN[i].shape[0] == input_feats[i].shape[0]
                    x_MNN_dict[i] = input_MNN[i][index_i, :]
                z_dict[i] = self.E_dict[i](x_dict[i])
                K_dict[i] = torch.exp(-torch.mean((x_dict[i].view(self.batch_size, 1, -1) - x_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                K_z_dict[i] = torch.exp(-torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = 0
                for i in range(num_datasets-1):
                    loss_D += (torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE = 0
            for i in range(num_datasets):
                loss_AE += torch.mean((self.G_dict[i](z_dict[i]) - x_dict[i])**2)

            # latent align loss:
            loss_LA = 0
            for i in range(num_datasets-1):
                loss_LA += torch.mean((z_dict[i] - self.E_dict[i+1](self.G_dict[i+1](z_dict[i])))**2)
                loss_LA += torch.mean((z_dict[i+1] - self.E_dict[i](self.G_dict[i](z_dict[i+1])))**2)

            # generator loss
            loss_G_GAN = 0
            for i in range(num_datasets-1):
                loss_G_GAN += -(torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()

            # geometric structure loss
            loss_Geo = 0
            for i in range(num_datasets):
                loss_Geo += - torch.clamp(cos(K_dict[i], K_z_dict[i]), max=0.975).mean()

            # MNN loss
            loss_MNN = 0
            for i in range(num_datasets-1):
                if input_MNN != None:
                    Sim = acquire_pairs(x_MNN_dict[i][:, feat_links_MNN[i][0]], 
                        x_MNN_dict[i+1][:, feat_links_MNN[i][1]], k=self.n_KNN)
                else:
                    Sim = acquire_pairs(x_dict[i][:, feat_links_MNN[i][0]], 
                        x_dict[i+1][:, feat_links_MNN[i][1]], k=self.n_KNN)
                Sim = torch.from_numpy(Sim).float().to(self.device)
                z_dist = torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i+1].view(1, self.batch_size, -1))**2, dim=2)
                loss_MNN += torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 200:
                print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
                 % (step, loss_D, loss_G_GAN, self.lambdaAE*loss_AE, self.lambdaGeo*loss_Geo, self.lambdaLA*loss_LA, self.lambdaMNN*loss_MNN))

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        for i in range(num_datasets):
            self.E_dict[i].train()
            # ✅ Handle sparse matrices in evaluation - convert to float32 to reduce VMS
            feat_data = input_feats[i]
            if sparse.issparse(feat_data):
                feat_data = feat_data.astype(np.float32).toarray()
            else:
                # Ensure float32 even if already dense
                feat_data = feat_data.astype(np.float32)
            z_dict[i] = self.E_dict[i](torch.from_numpy(feat_data).float().to(self.device))

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate([z_dict[i].detach().cpu().numpy() for i in range(num_datasets)], axis=0)


    def integrate_datasets_feats(self, # Use this function for N >= 3 datasets when provided linked features for MNN
                                 input_feats,
                                 paired_input_MNN, # In the form of [[link_feat_data1, link_feat_data2], ..., [link_feat_data(N_1), link_feat_dataN]]
                                 ):
        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))
        num_datasets = len(input_feats)
        self.E_dict = {}
        self.G_dict = {}
        params_G = []
        for i in range(num_datasets):
            self.E_dict[i] = encoder(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.E_dict[i].parameters()
            self.G_dict[i] = generator(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.G_dict[i].parameters()
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)

        self.D_dict = {}
        params_D = []
        for i in range(num_datasets-1):
            self.D_dict[i] = discriminator(self.n_latent).to(self.device)
            params_D += self.D_dict[i].parameters()
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.001)

        for i in range(num_datasets):
            self.E_dict[i].train()
            self.G_dict[i].train()
        for i in range(num_datasets-1):
            self.D_dict[i].train()

        # ✅ GPU memory monitoring setup
        device = None
        if torch.cuda.is_available():
            device = self.device if isinstance(self.device, int) else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
            print_gpu_memory("Before training loop ", device=device)
            print_cpu_memory("Before training loop ")
            print("-" * 70)

        try:
            import time as time_module
            training_start_time = time_module.time()
            
            for step in range(self.training_steps):
                # ✅ VERBOSE: Monitor every step near kill point (1900-2020) and every 10 steps otherwise
                verbose_zone = (step >= 1990 and step <= 2020) or (step % 10 == 0 and step < 120)
                
                # Track elapsed time
                elapsed_time = time_module.time() - training_start_time
                elapsed_minutes = elapsed_time / 60.0
                
                if verbose_zone:
                    print(f"\n{'='*70}")
                    print(f"→ Starting step {step}...")
                    print(f"⏱️  Elapsed time: {elapsed_minutes:.2f} minutes ({elapsed_time:.1f} seconds)")
                    print(f"🧵 step {step}: threads = {thread_count()}")
                    cpu_stats = get_cpu_memory_stats()
                    if cpu_stats:
                        print(f"📊 Memory BEFORE step {step}:")
                        print(f"   RSS: {cpu_stats['rss_gb']:.3f}GB | VMS: {cpu_stats['vms_gb']:.3f}GB | Shared: {cpu_stats['shared_gb']:.3f}GB")
                        print(f"   Process: {cpu_stats['process_memory_gb']:.3f}GB ({cpu_stats['process_percent']:.1f}%)")
                        print(f"   System Available: {cpu_stats['system_available_gb']:.1f}GB")
                        if cpu_stats['num_fds'] is not None:
                            print(f"   File descriptors: {cpu_stats['num_fds']}")
                        if cpu_stats['num_threads'] is not None:
                            print(f"   Threads: {cpu_stats['num_threads']}")
                        if hasattr(self, "_prev_cpu_memory_feats") and self._prev_cpu_memory_feats:
                            delta = cpu_stats['rss_gb'] - self._prev_cpu_memory_feats.get('rss_gb', 0)
                            if abs(delta) > 0.01:
                                print(f"   ΔRSS since last check: {delta:+.3f}GB")
                        
                        # Check for potential VMS-based OOM (some systems kill on VMS, not RSS)
                        if cpu_stats['vms_gb'] > 10.0:
                            print(f"   ⚠️  VMS is {cpu_stats['vms_gb']:.2f}GB - some systems kill on VMS limit!")
                            # ✅ NEW: Immediate VMS reduction
                            print(f"   🧹 Triggering immediate VMS cleanup...")
                            gc.collect()
                            malloc_trim()  # Critical: Return memory to OS
                
                # ✅ GPU memory monitoring at step 0 to catch OOM early
                if step == 0 and torch.cuda.is_available() and device is not None:
                    print_gpu_memory("Step 0 (before forward) ", device=device)
                
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                x_dict = {}
                z_dict = {}
                K_dict = {}
                K_z_dict = {}
                assert len(paired_input_MNN) == (num_datasets - 1)
                x_MNN_dict_0 = {}
                x_MNN_dict_1 = {}
                for i in range(num_datasets):
                    index_i = np.random.choice(np.arange(input_feats[i].shape[0]), size=self.batch_size)
                    
                    # ✅ Handle sparse matrices properly - convert to float32 to reduce VMS
                    batch_data = input_feats[i][index_i, :]
                    if sparse.issparse(batch_data):
                        batch_data = batch_data.astype(np.float32).toarray()
                    else:
                        # Ensure float32 even if already dense
                        batch_data = batch_data.astype(np.float32)
                    x_dict[i] = torch.from_numpy(batch_data).float().to(self.device)
                    
                    if i < (num_datasets-1):
                        x_MNN_dict_0[i] = paired_input_MNN[i][0][index_i, :]
                    if i > 0:
                        x_MNN_dict_1[i-1] = paired_input_MNN[i-1][1][index_i, :]
                    z_dict[i] = self.E_dict[i](x_dict[i])
                    K_dict[i] = torch.exp(-torch.mean((x_dict[i].view(self.batch_size, 1, -1) - x_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                    K_z_dict[i] = torch.exp(-torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                
                # ✅ VERBOSE: Memory after forward pass
                if verbose_zone:
                    cpu_stats = get_cpu_memory_stats()
                    if cpu_stats:
                        print(f"📊 Memory AFTER forward pass (step {step}): RSS={cpu_stats['rss_gb']:.3f}GB")

                # discriminator loss:
                for _ in range(5):
                    optimizer_D.zero_grad()
                    loss_D = 0
                    for i in range(num_datasets-1):
                        loss_D += (torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()
                    loss_D.backward(retain_graph=True)
                    optimizer_D.step()

                # autoencoder loss:
                loss_AE = 0
                for i in range(num_datasets):
                    loss_AE += torch.mean((self.G_dict[i](z_dict[i]) - x_dict[i])**2)

                # latent align loss:
                loss_LA = 0
                for i in range(num_datasets-1):
                    loss_LA += torch.mean((z_dict[i] - self.E_dict[i+1](self.G_dict[i+1](z_dict[i])))**2)
                    loss_LA += torch.mean((z_dict[i+1] - self.E_dict[i](self.G_dict[i](z_dict[i+1])))**2)

                # generator loss
                loss_G_GAN = 0
                for i in range(num_datasets-1):
                    loss_G_GAN += -(torch.log(1 + torch.exp(-self.D_dict[i](z_dict[i]))) + torch.log(1 + torch.exp(self.D_dict[i](z_dict[i+1])))).mean()

                # geometric structure loss
                loss_Geo = 0
                for i in range(num_datasets):
                    loss_Geo += - torch.clamp(cos(K_dict[i], K_z_dict[i]), max=0.975).mean()

                # MNN loss
                loss_MNN = 0
                Sim_tensors = []  # Store for cleanup after backward
                for i in range(num_datasets-1):
                    # ✅ acquire_pairs creates numpy array on CPU - convert to torch immediately
                    Sim_np = acquire_pairs(x_MNN_dict_0[i], x_MNN_dict_1[i], k=self.n_KNN)
                    Sim = torch.from_numpy(Sim_np).float().to(self.device)
                    del Sim_np  # Immediately delete numpy array to free CPU memory
                    z_dist = torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i+1].view(1, self.batch_size, -1))**2, dim=2)
                    loss_MNN += torch.sum(Sim * z_dist) / torch.sum(Sim)
                    Sim_tensors.append((Sim, z_dist))  # Keep for backward pass, cleanup later
                
                # ✅ VERBOSE: Memory after MNN computation
                if verbose_zone:
                    cpu_stats = get_cpu_memory_stats()
                    if cpu_stats:
                        print(f"📊 Memory AFTER MNN computation (step {step}): RSS={cpu_stats['rss_gb']:.3f}GB")

                optimizer_G.zero_grad()
                loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
                
                # ✅ GPU memory monitoring before backward pass (critical OOM point)
                if step == 0 and torch.cuda.is_available() and device is not None:
                    print_gpu_memory("Step 0 (before backward) ", device=device)
                    print_cpu_memory("Step 0 (before backward) ")
                
                # ✅ VERBOSE: Memory before backward (critical point)
                if verbose_zone:
                    cpu_stats = get_cpu_memory_stats()
                    if cpu_stats:
                        print(f"📊 Memory BEFORE backward (step {step}): RSS={cpu_stats['rss_gb']:.3f}GB")
                
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(params_G, 5.0)
                optimizer_G.step()
                
                # ✅ VERBOSE: Memory after backward
                if verbose_zone:
                    cpu_stats = get_cpu_memory_stats()
                    if cpu_stats:
                        print(f"📊 Memory AFTER backward (step {step}): RSS={cpu_stats['rss_gb']:.3f}GB")
                
                # ✅ CRITICAL: Log before deleting tensors (so loss variables still exist)
                # Store loss values as Python floats for logging after cleanup
                loss_D_val = float(loss_D.item() if hasattr(loss_D, 'item') else loss_D)
                loss_G_GAN_val = float(loss_G_GAN.item() if hasattr(loss_G_GAN, 'item') else loss_G_GAN)
                loss_AE_val = float(loss_AE.item() if hasattr(loss_AE, 'item') else loss_AE)
                loss_Geo_val = float(loss_Geo.item() if hasattr(loss_Geo, 'item') else loss_Geo)
                loss_LA_val = float(loss_LA.item() if hasattr(loss_LA, 'item') else loss_LA)
                loss_MNN_val = float(loss_MNN.item() if hasattr(loss_MNN, 'item') else loss_MNN)
                
                if not step % 2000:
                    print("step %d, loss_D=%f, loss_GAN=%f, loss_AE=%f, loss_Geo=%f, loss_LA=%f, loss_MNN=%f"
                     % (step, loss_D_val, loss_G_GAN_val, self.lambdaAE*loss_AE_val, self.lambdaGeo*loss_Geo_val, self.lambdaLA*loss_LA_val, self.lambdaMNN*loss_MNN_val))
                    if torch.cuda.is_available() and device is not None:
                        print_gpu_memory(f"Step {step} ", device=device, print_peak=True)
                    print_cpu_memory(f"Step {step} ")
                
                # ✅ VERBOSE: Monitor every 10 steps for first 100 steps to catch kill point
                if step % 10 == 0 and step < 100 and step > 0:
                    if torch.cuda.is_available() and device is not None:
                        print_gpu_memory(f"Step {step} ", device=device)
                    print_cpu_memory(f"Step {step} ")
                    print(f"  loss_D={loss_D_val:.4f}, loss_AE={self.lambdaAE*loss_AE_val:.2f}, loss_MNN={self.lambdaMNN*loss_MNN_val:.2f}")
                
                # ✅ CRITICAL: Aggressive cleanup after backward pass to prevent memory accumulation
                # Delete all intermediate tensors that accumulate over steps
                if verbose_zone:
                    cpu_before_cleanup = get_cpu_memory_stats()
                    if cpu_before_cleanup:
                        print(f"📊 Memory BEFORE cleanup (step {step}): RSS={cpu_before_cleanup['rss_gb']:.3f}GB")
                
                for Sim, z_dist in Sim_tensors:
                    del Sim, z_dist
                del Sim_tensors
                
                # Delete pairwise distance matrices (memory-intensive)
                for i in range(num_datasets):
                    if i in K_dict:
                        del K_dict[i]
                    if i in K_z_dict:
                        del K_z_dict[i]
                del K_dict, K_z_dict
                
                # Delete input/output tensors
                for i in range(num_datasets):
                    if i in x_dict:
                        del x_dict[i]
                    if i in z_dict:
                        del z_dict[i]
                del x_dict, z_dict
                
                # Delete MNN input dictionaries
                for i in range(num_datasets-1):
                    if i in x_MNN_dict_0:
                        del x_MNN_dict_0[i]
                    if i in x_MNN_dict_1:
                        del x_MNN_dict_1[i]
                del x_MNN_dict_0, x_MNN_dict_1
                
                # Delete loss tensors
                del loss_G, loss_AE, loss_LA, loss_G_GAN, loss_Geo, loss_MNN
                
                # ✅ VERBOSE: Memory after cleanup
                if verbose_zone:
                    cpu_after_cleanup = get_cpu_memory_stats()
                    if cpu_after_cleanup and cpu_before_cleanup:
                        delta = cpu_after_cleanup['rss_gb'] - cpu_before_cleanup['rss_gb']
                        print(f"📊 Memory AFTER cleanup (step {step}): RSS={cpu_after_cleanup['rss_gb']:.3f}GB (Δ{delta:+.3f}GB)")
                    # Store for next step delta calculation
                    if cpu_after_cleanup:
                        self._prev_cpu_memory_feats = cpu_after_cleanup.copy()
                    print(f"{'='*70}\n")
                else:
                    # Still store memory state even if not verbose for delta tracking
                    cpu_after_cleanup = get_cpu_memory_stats()
                    if cpu_after_cleanup:
                        self._prev_cpu_memory_feats = cpu_after_cleanup.copy()
                
                # ✅ GPU memory monitoring after backward pass
                if step == 0 and torch.cuda.is_available() and device is not None:
                    print_gpu_memory("Step 0 (after backward) ", device=device, print_peak=True)
                    print("-" * 70)
                
                # ✅ Periodic cleanup to prevent gradual memory leaks
                if step % 10 == 0 and step > 0:
                    # NEW: Ultra-frequent malloc_trim to keep VMS low
                    malloc_trim()
                if step % 25 == 0 and step > 0:
                    # Light cleanup - CUDA cache + malloc_trim
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    malloc_trim()  # Add here too
                    if verbose_zone:
                        print(f"🧹 Light cleanup at step {step} (CUDA cache)")
                if step % 100 == 0 and step > 0:
                    # Aggressive cleanup - both CPU and GPU
                    if verbose_zone:
                        cpu_before_gc = get_cpu_memory_stats()
                        if cpu_before_gc:
                            print(f"🧹 Aggressive cleanup at step {step}: RSS={cpu_before_gc['rss_gb']:.3f}GB, VMS={cpu_before_gc['vms_gb']:.3f}GB")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    # ✅ CRITICAL: Return freed memory to OS to reduce VMS
                    malloc_trim()
                    if verbose_zone:
                        cpu_after_gc = get_cpu_memory_stats()
                        if cpu_after_gc and cpu_before_gc:
                            rss_delta = cpu_after_gc['rss_gb'] - cpu_before_gc['rss_gb']
                            vms_delta = cpu_after_gc['vms_gb'] - cpu_before_gc['vms_gb']
                            print(f"   After GC+trim: RSS={cpu_after_gc['rss_gb']:.3f}GB (Δ{rss_delta:+.3f}GB), VMS={cpu_after_gc['vms_gb']:.3f}GB (Δ{vms_delta:+.3f}GB)")
                
                # ✅ Monitor CPU memory growth to catch OOM before it happens
                if step % 500 == 0 and step > 0:
                    current_cpu_memory = get_cpu_memory_stats()
                    if current_cpu_memory:
                        if hasattr(self, "_prev_cpu_memory_feats"):
                            cpu_growth = current_cpu_memory["process_memory_gb"] - self._prev_cpu_memory_feats["process_memory_gb"]
                            if cpu_growth > 0.5:  # If process grew by more than 0.5GB
                                print(f"Step {step}: ⚠️ CPU memory growth +{cpu_growth:.2f}GB since last check")
                                print(f"    Process using {current_cpu_memory['process_memory_gb']:.2f}GB, forcing GC...")
                                gc.collect()
                        self._prev_cpu_memory_feats = current_cpu_memory.copy()
                        
                        # ✅ NEW: Check VMS growth - this is what triggers the kill!
                        if current_cpu_memory['vms_gb'] > 10.0:  # VMS over 10GB
                            print(
                                f"⚠️ WARNING: VMS is {current_cpu_memory['vms_gb']:.2f}GB - triggering aggressive cleanup!"
                            )
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            malloc_trim()  # CRITICAL: Return memory to OS to reduce VMS
                        
                        # Emergency cleanup if system RAM is low
                        if current_cpu_memory['system_available_gb'] < 3.0:
                            print(f"⚠️ CRITICAL: System RAM low ({current_cpu_memory['system_available_gb']:.2f}GB available)!")
                            print("    Forcing emergency cleanup...")
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            malloc_trim()  # Return memory to OS
                
                # ✅ Check for time-based limits (common in job schedulers)
                if step % 1000 == 0 and step > 0:
                    elapsed_hours = elapsed_minutes / 60.0
                    print(f"\n⏱️  Progress check at step {step}: {elapsed_hours:.2f} hours elapsed")
                    print(f"   Average: {elapsed_time/step:.3f} seconds/step")
                    print(f"   Estimated remaining: {(self.training_steps - step) * elapsed_time/step / 3600:.2f} hours")
                    
                    # Check for common time limits
                    if elapsed_hours > 20:
                        print(f"   ⚠️  Running for >20 hours - might hit walltime limit!")
                    if elapsed_hours > 23:
                        print(f"   ⚠️  CRITICAL: Running for >23 hours - likely to hit 24h walltime!")
        
        except Exception as e:
            print("\n" + "="*70)
            print("EXCEPTION CAUGHT DURING TRAINING!")
            print("="*70)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available() and device is not None:
                print_gpu_memory("At exception ", device=device, print_peak=True)
            print_cpu_memory("At exception ")
            print("="*70)
            raise

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        begin_time = time.time()
        print("Begining time: ", time.asctime(time.localtime(begin_time)))

        for i in range(num_datasets):
            self.E_dict[i].train()
            # ✅ Handle sparse matrices in evaluation - convert to float32 to reduce VMS
            feat_data = input_feats[i]
            if sparse.issparse(feat_data):
                feat_data = feat_data.astype(np.float32).toarray()
            else:
                # Ensure float32 even if already dense
                feat_data = feat_data.astype(np.float32)
            z_dict[i] = self.E_dict[i](torch.from_numpy(feat_data).float().to(self.device))

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate([z_dict[i].detach().cpu().numpy() for i in range(num_datasets)], axis=0)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {}
        for i in range(num_datasets):
            state['E_%d' % i] = self.E_dict[i].state_dict()
            state['G_%d' % i] = self.G_dict[i].state_dict()

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))
