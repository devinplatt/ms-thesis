"""
This script creates TFRecords from FMA mp3 files.
To run more quickly, it uses multiprocessing.
To save space, it uses ZLIB compression.

Due to laziness, the directory with FMA audio files (fma_large_dir) is a
hard-coded string rather than a command line argument.

To run this script, one needs to download the fma_large.zip file, linked to at
https://github.com/mdeff/fma
and then point fma_large_dir to the unzipped directory.

The output of this script a saved to tf_record_shard_dir, which also needs to be
replaced with a real directory name.
"""
from collections import Counter, defaultdict
import csv
import h5py
import json
import librosa
import multiprocessing
import numpy as np
import os
import random
import tensorflow as tf
from skdata.mnist.views import OfficialVectorClassification
from tqdm import tqdm
import datetime
# Verify that protobuf implementation is C++, not Python.
from google.protobuf.internal import api_implementation

print('default protobuf implementation: {}'.format(
      api_implementation._default_implementation_type)
)
print('protobuf implementation: {}'.format(
      api_implementation.Type())
)
st = datetime.datetime.now()
print('Getting the FMA/LFM-1b matching subset!')
fma_matched_dir = '../matchings/fma_lfm-1b'
matched_fma_track_ids_fname = os.path.join(fma_matched_dir,
                                           'artist_trackname_to_fma_ids.txt')
matched_artists_tracks_fname = os.path.join(fma_matched_dir,
                                            'matched_artists_tracks.txt')
matched_artists_tracks_tuples_list = [
    tuple(line.strip().split('\t'))
    for line in open(matched_artists_tracks_fname)
]
matched_fma_track_ids = [
    tuple(line.strip().split('\t'))
    for line in open(matched_fma_track_ids_fname)
]
fma_track_id_to_matched_index = {
    track_id: index
    for index, track_ids in enumerate(matched_fma_track_ids)
    for track_id in track_ids
}
artist_trackname_to_fma_track_ids = {
    '\t'.join(at): track_ids
    for at, track_ids in zip(matched_artists_tracks_tuples_list,
                             matched_fma_track_ids)
}
print('Getting the matching of artist_trackname to matrix index.')
matrix_artist_tracknames_fname = '../matchings/both/matched_artists_tracks.txt'
matrix_artist_tracknames = [
    line.strip() for line in open(matrix_artist_tracknames_fname)
]
artist_trackname_to_matrix_index = {
    artist_trackname: index
    for index, artist_trackname in enumerate(matrix_artist_tracknames)
}
print('Getting song factors.')
song_factors_fname = '../latent_factors/output/factors_merged_38_v.npy'
song_factors = np.load(song_factors_fname)


def fma_track_id_to_fname(fma_id):
    # Download the fma_large.zip file, linked to at
    # https://github.com/mdeff/fma
    # Then unzip it and provide it's path here.
    fma_large_dir = '/your/path/to/fma_large'
    mp3_fname_template = '{three_digit}/{six_digit}.mp3'
    fma_id = int(fma_id)
    three_digit = str(fma_id / 1000).zfill(3)
    six_digit = str(fma_id).zfill(6)
    return os.path.join(fma_large_dir,
                        mp3_fname_template.format(three_digit=three_digit,
                                                  six_digit=six_digit)
                        )


def get_latent_factors(i):
    artist, track_name = matched_artists_tracks_tuples_list[i]
    artist_trackname = '\t'.join([artist, track_name])
    latent_factor_index = artist_trackname_to_matrix_index[artist_trackname]
    latent_factor = song_factors[latent_factor_index]
    return latent_factor

sample_rate = 16000
duration_seconds = 20 # 29
target_num_samples = duration_seconds * sample_rate


def load_audio_file(fname):
    """
    Loads raw audio. Ensures that audio is fixed length.
    """
    audio, _ = librosa.load(fname, sr=sample_rate)  # whole signal
    num_samples = audio.shape[0]
    # If too short, pad with zeros.
    if num_samples < target_num_samples:  
        audio = np.hstack((audio, np.zeros((target_num_samples - num_samples,))))
    # If too long, pick center audio of length target_num_samples.
    elif num_samples > target_num_samples:
        audio = audio[
            (num_samples-target_num_samples)/2:(num_samples+target_num_samples)/2
        ]
    return audio


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecords_file_from_track_ids(track_ids, tfrecords_fname):
    trIdx = range(len(track_ids))
    random.shuffle(trIdx)
    # From: https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    # One MUST randomly shuffle data before putting it into one of these
    # formats. Without this, one cannot make use of tensorflow's great
    # out of core shuffling.
    writer = tf.python_io.TFRecordWriter(
        tfrecords_fname,
        tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.ZLIB
            )
        )
    for example_idx in trIdx:
        fma_track_id = track_ids[example_idx]
        matched_index = fma_track_id_to_matched_index[fma_track_id]
        artist, track_name = matched_artists_tracks_tuples_list[matched_index]
        latent_factors = get_latent_factors(matched_index)
        try:
            audio_fname = fma_track_id_to_fname(fma_track_id)
            audio = load_audio_file(audio_fname)
        except Exception as e:
            # print(e)
            continue
        # construct the Example proto object
        example = tf.train.Example(
            # Example contains a Features proto object
            features = tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature = {
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'factors': tf.train.Feature(
                        float_list=tf.train.FloatList(value=latent_factors)
                    ),
                    'audio': tf.train.Feature(
                        float_list = tf.train.FloatList(
                            value=audio  # .astype("float32")
                        )
                    ),
                    'fma_track_id': _bytes_feature(tf.compat.as_bytes(fma_track_id)),
                    'artist': _bytes_feature(tf.compat.as_bytes(artist)),
                    'track_name': _bytes_feature(tf.compat.as_bytes(track_name))
                }
            )
        )
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)
    writer.close()


def create_tfrecords_file(map_tuple):
    artist_trackname_fname, tfrecords_fname = map_tuple
    artist_tracknames = [line.strip() for line in open(artist_trackname_fname)]
    track_ids_lists = [
        artist_trackname_to_fma_track_ids[at] for at in artist_tracknames
    ]
    # We pick just the first track id of matched track ids.
    # This choice is arbitrary, but we just need one track id.
    track_ids = [t[0] for t in track_ids_lists]
    create_tfrecords_file_from_track_ids(track_ids, tfrecords_fname)

et = datetime.datetime.now()
print('Setup took: {}'.format(str(et - st)))
shard_parent_dir = '/home/devin/git/ms-thesis/split/fma/shards'
shard_dirs = [
    os.path.join(shard_parent_dir, dirname)
    for dirname in os.listdir(shard_parent_dir)
]
shard_files = [
    [os.path.join(shard_dir, fname) for fname in os.listdir(shard_dir)]
    for shard_dir in shard_dirs
]
all_shard_files = [
    fname
    for fnames in shard_files
    for fname in fnames
]
tf_record_shard_dir = '/path/to/your/output/'
shard_tfrecord_files = [
    os.path.join(tf_record_shard_dir,
                 fname.split('/')[-2] + '/' + fname.split('/')[-1] + '_zlib.tfrecord')
    for fname in all_shard_files
]
input_output_tuples = [
    (ifname, ofname)
    for ifname, ofname in zip(all_shard_files, shard_tfrecord_files)
]
random.shuffle(input_output_tuples)
tfrecord_dirs = [
    os.path.join(tf_record_shard_dir, dirname)
    for dirname in os.listdir(tf_record_shard_dir)
]
existing_tfrecord_files = [
    [os.path.join(tfrecord_dir, fname) for fname in os.listdir(tfrecord_dir)]
    for tfrecord_dir in tfrecord_dirs
]
existing_fnames = set(
    fname for fnames in existing_tfrecord_files
    for fname in fnames
)
old_number_fnames_todo = len(input_output_tuples)
input_output_tuples = filter(lambda x: x[1] not in existing_fnames,
                             input_output_tuples)
new_number_fnames_todo = len(input_output_tuples)
print(
    'There are {} total tfrecord shards to create.'.format(
        len(all_shard_files)
    )
)
print(
    'We have already done {} tfrecords, so we ignore those.'.format(
        old_number_fnames_todo - new_number_fnames_todo
    )
)
print('Generating {} shards'.format(len(input_output_tuples)))
print(
    'Estimated time to completion: {} minutes'.format(
        len(input_output_tuples) * 1000 * .4 / float(60)
    )
)
st = datetime.datetime.now()
# Use map instead of multiprocessing.Pool().map when debugging.
# map(create_tfrecords_file, input_output_tuples)
p = multiprocessing.Pool(4)
p.map(create_tfrecords_file, input_output_tuples)
et = datetime.datetime.now()
print('Creating TFRecords file shards took: {}'.format(str(et - st)))
