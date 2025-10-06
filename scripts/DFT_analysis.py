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

activations_path = "/large_storage/hielab/changdan/evo2_gtdb_complete_activations_sae-a6u40nnl-layer26-batch-topk-tied-expansion_8-k_64/"
fastas_path = "/large_storage/hielab/changdan/evo_sae_1_20_25_prokaryotic_dataset_final_permission_fixed/gtdb_rep_genomes_complete/"

contig_dict = {}

chunk_size = 50_000

for f in tqdm(os.listdir(fastas_path)[:1]):
    fasta_name = f[:-4]
    contig_list = list(SeqIO.parse(fastas_path + f, 'fasta'))

    for contig_record in contig_list:
        # sparse = scipy.sparse.load_npz(f"{activations_path}{fasta_name}/{contig_record.id}.npz")
        sparse_chunks = []
        for start in range(0, len(contig_record), chunk_size):
            end = min(start + chunk_size, len(contig_record))
            sparse_chunks.append(scipy.sparse.load_npz(f"{activations_path}{fasta_name}/{contig_record.id}_{start}_{end}.npz"))
        sparse = scipy.sparse.vstack(sparse_chunks)
            
        contig_dict[(fasta_name, contig_record.id)] = sparse




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


arr_all = contig_dict[('GCF_009646115.1',
  'NZ_CP045720.1')][:]

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

np.save("../output/NZ_CP045720.1_peak_periods.npy", peak_periods)
np.save("../output/NZ_CP045720.1_peak_magnitudes.npy", peak_magnitudes)

