# latent factors
Instructions on how to generate the latent factors will be placed here.
We assume that the generated factors will be saved in the "output" directory.
Code in the "eval" directory will allow for evaluation of the latent factors.

I've computed a matrix of a subset of user listening events from the LastFM-1b dataset (using the ms-thesis/latent_factors/create_lfm_1b_interactions_matrix_merged.py script). This matrix is linked to the Million Song Dataset and FMA tracks using the included matching files, which correspond by line number. Each song column in the listening event matrix corresponds to the lfm-1b track id in the corresponding row of matching/both/artist_trackname_to_lfm_1b_ids.txt.

Unfortunately, I cannot include the matrix in the repository because of its large size (over 800MB). If you wish to compute a listening events matrix for yourself you need to download the LFM-1b dataset yourself (from http://www.cp.jku.at/datasets/LFM-1b/, 8GB zipped). 

Fortunately, I CAN provide latent factors for the tracks in FMA and MSD (<100 MB if using 38 factors, see latent_factors/output/factors_merged_38_v.npy). These were computed using the ms-thesis/latent_factors/wmf/run_wmf.py script, which runs weighted matrix factorization on the listening events matrix. The script uses code from Sander Dieleman (https://github.com/benanne/wmf).

