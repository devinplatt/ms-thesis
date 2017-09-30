#!/usr/bin/env python
"""
Script for running WMF.
"""
import argparse
import datetime
import numpy as np
import numpy
import os
import pandas
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

import wmf

parser = argparse.ArgumentParser(description='Train a WMF model')
parser.add_argument('--binary',
                    action='store_true',
                    help='If set, we convert the input matrix to binary 0-1 playcounts.')
parser.add_argument('--num_factors',
                    type=int,
                    default=40,
                    help='Number of latent factors to model.')
parser.add_argument('--num_iterations',
                    type=int,
                    default=1,
                    help='Number of iterations to run.')
parser.add_argument('--batch_size',
                    type=int,
                    default=1000,
                    help='Size of each batch sent to GPU.')
parser.add_argument('--input_matrix_fname',
                    default='/home/devin/git/ms-thesis/latent_factors/output/LastFM-1b_matrix_merged.npz',
                    help='Input listening events matrix to load and run WMF on.')
parser.add_argument('--alpha',
                    type=float,
                    default=2.0,
                    help='Alpha used for confidences.')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-6,
                    help='Alpha used for confidences.')
parser.add_argument('--save_path',
                    default='/home/devin/git/ms-thesis/latent_factors/output/factors_merged',
                    help='Where to save the factors.')
args = parser.parse_args()
binary = args.binary
num_factors = args.num_factors
num_iterations = args.num_iterations
# 10000 means 200 MB of GPU mem used, mem usage scales ~linearly
batch_size = args.batch_size
input_matrix_fname = args.input_matrix_fname
alpha = args.alpha
epsilon = args.epsilon
save_path = args.save_path

# Weird Cuda error: "OSError: cusolver library not found"
# Solved with the following code, taken from:
#    http://stackoverflow.com/questions/38263085/cusolver-library-not-found
#    https://github.com/lebedov/scikit-cuda/issues/171
import ctypes
# This hardcoded file name was found using
#     locate libcusolver.so
_libcusolver_libname = '/usr/local/cuda-8.0/lib64/libcusolver.so'
ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
_libcusolver = ctypes.cdll.LoadLibrary(_libcusolver_libname)
from skcuda import cusolver
import batched_inv
import batched_inv_precompute
import solve_mp
import solve_gpu


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
           
                         shape = loader['shape'])

class TopRelated(object):
    def __init__(self, artist_factors):
        # fully normalize artist_factors, so can compare with only the dot product
        norms = numpy.linalg.norm(artist_factors, axis=-1)
        self.factors = artist_factors / norms[:, numpy.newaxis]

    def get_related(self, artistid, N=10):
        scores = self.factors.dot(self.factors[artistid])
        best = numpy.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])


print('Loading listening events matrix.')
st = datetime.datetime.now()
mat = load_sparse_csr(input_matrix_fname)
if binary:
    print('converting matrix to binary 0-1 playcounts')
    mat = (mat > 0.5).astype(np.int64)
et = datetime.datetime.now()
print("Loading dataset took: {}".format(str(et - st)))

print('Computing confidences')
st = datetime.datetime.now()
S = wmf.log_surplus_confidence_matrix(mat, alpha=2.0, epsilon=1e-6)
et = datetime.datetime.now()
print("Computing confidences took: {}".format(str(et - st)))

print('Running WMF algorithm.')
st = datetime.datetime.now()
solve = solve_gpu.solve_gpu
U, V = \
    wmf.factorize(
        S,
        num_factors=num_factors,
        lambda_reg=1e-5,
        num_iterations=num_iterations,
        init_std=0.01,
        verbose=True,
        dtype='float32',
        recompute_factors=batched_inv_precompute.recompute_factors_bias_batched_precompute,
        batch_size=batch_size,
        solve=solve)
et = datetime.datetime.now()
print("Running WMF algorithm took: {} for {} iterations".format(
      str(et - st),
      num_iterations)
)

print('Saving Factors.')
u_path = save_path + '_u.npy'
v_path = save_path + '_v.npy'
np.save(u_path, U)
np.save(v_path, V)
print('Done saving Factors.')
