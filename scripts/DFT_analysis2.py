import pandas as pd
import numpy as np
import scipy
import os
from Bio import SeqIO, SearchIO
import math
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from typing import List, Optional, Callable






genome_path_t2t = "/large_storage/hielab/changdan/euk_genomes/GCF_009914755.1_T2T-CHM13v2.0_genomic.gbff"
activations_path_t2t = "/large_storage/hielab/changdan/evo2_T2T_activations_sae-a6u40nnl-layer26-batch-topk-tied-expansion_8-k_64/"

fa = SeqIO.parse(genome_path_t2t, "genbank")
chroms = [g for g in fa]
chunk_size = 50000

def load_contig(contig, num_chunks=None):
    chunks = []

    i = 0
    for start in tqdm(range(0, len(contig), chunk_size)):
        i += 1
        if num_chunks is not None and i > num_chunks: break
        stop = min(start + chunk_size, len(contig))
        sparse = scipy.sparse.load_npz(f"{activations_path_t2t}{contig.id}/{contig.id}_{start}_{stop}.npz")
        chunks.append(sparse)

    chunks_cat = scipy.sparse.vstack(chunks)

    return chunks_cat
activations_human = load_contig(chroms[19], 200)










import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft

def analyze_dft(arr, sampling_rate=1.0, plot=True, title='DFT Analysis'):
    # Flatten the input array if it's 2D
    if arr.ndim == 2:
        arr_1d = arr.flatten()
    else:
        arr_1d = arr.copy()
    
    n = len(arr_1d)
    
    # Compute DFT
    dft = np.fft.fft(arr_1d)
    
    # Compute frequency bins
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    
    magnitude = np.abs(dft)
    
    positive_freqs = freqs[:n//2]
    positive_magnitude = magnitude[:n//2]
    
    mask = positive_freqs > 0.1
    
    peak_period = (1 / positive_freqs[mask])[positive_magnitude[mask].argmax()]
    peak_magnitude = positive_magnitude[mask].max()

    return peak_period, peak_magnitude

    





        
        
from joblib import Parallel, delayed
import numpy as np


arr_all = activations_human

def process_feature(i, arr_all):
    feature_arr = np.array(arr_all[:, i].todense())
    peak_period, peak_magnitude = analyze_dft(feature_arr)
    return peak_period, peak_magnitude

# Process in parallel with progress bar
results = Parallel(n_jobs=128)(
    delayed(process_feature)(i, arr_all) 
    for i in tqdm(range(32768), desc="Processing features")
    # for i in tqdm(range(500), desc="Processing features")
)

# Unpack results
peak_periods, peak_magnitudes = zip(*results)

peak_periods = np.array(peak_periods)
peak_magnitudes = np.array(peak_magnitudes)

np.save("../output/NC_060944.1_peak_periods.npy", peak_periods)
np.save("../output/NC_060944.1_peak_magnitudes.npy", peak_magnitudes)

