#!/bin/bash
mkdir -p shards/train
shuf train_artist_tracknames.txt | split --lines=1000 --numeric-suffixes --suffix-length=3 - shards/train/shard_

mkdir -p shards/test
split --lines=1000 --numeric-suffixes --suffix-length=3 test_artist_tracknames.txt shards/test/shard_

mkdir -p shards/valid
split --lines=1000 --numeric-suffixes --suffix-length=3 valid_artist_tracknames.txt shards/valid/shard_
