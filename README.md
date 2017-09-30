# ms-thesis

This repository is provided so that anyone can repeat experiments similar to those done in my Master's thesis.

In my thesis I used audio clips from the Million Song Dataset, which are not generally available to the public. In this repository, I have included instructions using the smaller, but easily downloaded, Free Music Archive dataset.

I've computed a matrix of a subset of user listening events from the LastFM-1b dataset (using the ms-thesis/latent_factors/create_lfm_1b_interactions_matrix_merged.py script). This matrix is linked to the Million Song Dataset and FMA tracks using the included matching files, which correspond by line number. Each song column in the listening event matrix corresponds to the lfm-1b track id in the corresponding row of matching/both/artist_trackname_to_lfm_1b_ids.txt.

Unfortunately, I cannot include the matrix in the repository because of its large size (over 800MB). If you wish to compute a listening events matrix for yourself you need to download the LFM-1b dataset yourself (from http://www.cp.jku.at/datasets/LFM-1b/, 8GB zipped). 

Fortunately, I CAN provide latent factors for the tracks in FMA and MSD (<100 MB if using 38 factors, see latent_factors/output/factors_merged_v.npy). These were computed using the ms-thesis/latent_factors/wmf/run_wmf.py script, which runs weighted matrix factorization on the listening events matrix. The script uses code from Sander Dieleman (https://github.com/benanne/wmf).

A training/validation/test split can be found in the split/ directory. I split the data 90/10 for training/test, with 10% of the training data reserved for validation (thus a 82/8/10 train/validation/test split).

Jupyter Notebooks containing evaluations of the latent factors can be found in the latent_factor/eval directory. Some notebooks include qualitative evaluation of the factors ("does dot-product similarity result in similar songs?"), while other notebooks train a logistic regression model to predict musical tags from latent factors.
