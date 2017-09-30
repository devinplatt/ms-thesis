# We prepare the user-track matrix from LastFM-1b data.
# For the MSD and FMA matched tracks, this takes about 30 minutes to run on
# my machine, and the output is about 1 GB in size.
#
# This script uses merged lfm-1b track ids (merged by having the
# same artist and track title.)
import argparse
from collections import Counter, defaultdict
import datetime
import os
import numpy as np
import scipy
from scipy import sparse

parser = argparse.ArgumentParser(description='Convert matrix to binary.')
parser.add_argument('--lfm_1b_ids_fname',
                    default='/home/devin/git/ms-thesis/matchings/both/artist_trackname_to_lfm_1b_ids.txt',
                    help='Name of file containing tab-separated lists of desired lfm-1b ids, line-by-line.')
parser.add_argument('--lfm_1b_dir',
                    default='/home/devin/data/social/lfm-1b/',
                    help='Directory containing lfm-1b.')
parser.add_argument('--num_lines',
                    type=int,
                    default=None,
                    help='Limit on the number of lines of the LE file to process (for time profiling).')
parser.add_argument('--save_path',
                    default='/home/devin/git/ms-thesis/latent_factors/output/LastFM-1b_matrix_merged.npz',
                    help='Where to save the factors.')
args = parser.parse_args()


# Save sparse matrix.
# From: http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


# How to load this matrix, for reference.
def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix(
                                   (
                                    loader['data'],
                                    loader['indices'],
                                    loader['indptr']
                                    ),
                                   shape = loader['shape']
                                   )
# We define a matrix for just the matched tracks, so we need
# to impose and index order in the matrix on those tracks.
# Naturally, we use the order of those tracks in the track_ids
# file.  
print('Getting tracks.')
track_ids_fname = args.lfm_1b_ids_fname

print('Making LFM-1b track_id -> matrix_index many-to-one mapping.')
lfm_1b_track_ids = [[int(x) for x in line.strip().split('\t')] for line in open(track_ids_fname)]
lfm_1b_track_id_to_matrix_index = {
    lfm_1b_id: index
    for index, lfm_1b_ids in enumerate(lfm_1b_track_ids)
    for lfm_1b_id in lfm_1b_ids
}

track_ids_set = set(lfm_1b_id for lfm_1b_ids in lfm_1b_track_ids for lfm_1b_id in lfm_1b_ids)
print(len(track_ids_set))
print(list(track_ids_set)[:5])

# We define a matrix for all the users in the LFM-1b dataset,
# (even if by chance they haven't played any of the tracks in our subset.)
# So we need to impose matrix indices for each user. Naturally
print('Getting LFM-1b user_id to matrix user index mapping.')
lfm_1b_dir = args.lfm_1b_dir
users_fname = os.path.join(lfm_1b_dir, 'LFM-1b_users.txt')
# We remember to get rid of the header below.
users = [int(line.strip().split('\t')[0]) for line in open(users_fname)
         if line.strip().split('\t')[0] != 'user_id']
lfm_1b_user_id_to_matrix_index = {
    user_id: index for index, user_id in enumerate(users)
}

# The format of the listening event (le) file is
# user-id, artist-id, album-id, track-id, timestamp
# eg.
# 31435741    2    4    4    1385212958
#
# See: http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf
le_fname = os.path.join(lfm_1b_dir, 'LFM-1b_LEs.txt')
# We use lil_matrix, since it is much faster to add new entries.
user_track_matrix = scipy.sparse.lil_matrix(
                                            (
                                             len(users), 
                                             len(lfm_1b_track_ids)
                                            )
)

print('Running through the giant Listening Event (LE) file.')
st = datetime.datetime.now()
with open (le_fname) as le_file:
    ctr = 0
    for line in le_file:
        # Read and parse the line.
        split_line = line.strip().split('\t')
        track_id = int(split_line[3])
        user_num = int(split_line[0])

        ctr +=1

        if ctr == 1:
            current_user = user_num
            track_index_to_count = defaultdict(int)

        if ctr % 10000000 == 0:
            et = datetime.datetime.now()
            print('Took: {} for {} lines'.format(str(et -st), ctr))
            print(ctr)

        # Early stopping for testing and time profiling.
        if args.num_lines and ctr == args.num_lines:
            print('Halting for maximum number of lines reached: {}'.format(args.num_lines))
            break

        if current_user != user_num:
            # We've swtiched users. Flush the user row to the matrix,
            # and reset the user.

            # Flush user row!
            if current_user in lfm_1b_user_id_to_matrix_index:
                user_index = lfm_1b_user_id_to_matrix_index[current_user]
                for track_index, plays in track_index_to_count.iteritems():
                    user_track_matrix[user_index, track_index] = plays

            current_user = user_num
            track_index_to_count = defaultdict(int)
        
        # We include plays of tracks in our subset.
        # Note that with our normalization by capitalization,
        # there are duplicate tracks which need to be ignored.
        if track_id in track_ids_set:
            track_index = lfm_1b_track_id_to_matrix_index[track_id]
            track_index_to_count[track_index] += 1

et = datetime.datetime.now()
print('Took: {} for {} lines'.format(str(et -st), ctr))

# Now add the final user to the matrix, and save the results...
# Adding final user...
user_index = lfm_1b_user_id_to_matrix_index[current_user]
for track_index, plays in track_index_to_count.iteritems():
    user_track_matrix[user_index, track_index] = plays

print('Saving Results!')
output_matrix_fname = args.save_path
user_track_matrix = user_track_matrix.tocsr()
save_sparse_csr(output_matrix_fname, user_track_matrix)

