# Neural Architecture Search (NAS)

This code repository is for the reproducibility of NeurIPS2020 submission paper "Differentiable Neural Architecture Search in Equivalent Space with Exploration Enhancement".

This code is built based on "NAS-Bench-201" dataset, please refer NAS-Bench-201.md for the use of NAS-Bench-201.

Our proposed EENAS needs to off-line train a DVAE(graph neural network based autoencoder), and we implement it based on "D-VAE: A Variational Autoencoder for Directed Acyclic Graphs, Advances in Neural Information Processing Systems (NeurIPS-19)." https://github.com/muhanzhang/D-VAE. We put our trained autoencoder in './exps.algos.data_dvae', where we use a 3-layerl linear enural network to fit the decoder in the original DVAE.

Our EENAS model could be found in './lib/models/search_model_eenas.py'.

The genral algorithm is in './exps/algos/EENAS_xxx.ipynb'

and the comparison results are also in those folds './exps/algos/EENAS_results'
