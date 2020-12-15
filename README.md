# Recommendation System

## Topic: Variational Autoencoders for Collaborative Filtering
> Team: Tuan BM, Quan VD, Luong NV, Dang NM

-------------------------------------------------------------------------------------

## Data

  + Main:
      + [MovieLens-25M](http://files.grouplens.org/datasets/movielens/ml-25m.zip)
  + Others: 
      + [MovieLens-20M](http://files.grouplens.org/datasets/movielens/ml-20m.zip)
      + [MovieLens-10M](http://files.grouplens.org/datasets/movielens/ml-10m.zip)


## Methods

> We define two related models: denoising autoencoder with multinomial likelihood (Multi-DAE in the paper) and partially-regularized variational autoencoder with multinomial likelihood (Multi-VAE^{PR} in the paper).

### Multi DAE
  
__Notations__: We use <img src="https://render.githubusercontent.com/render/math?math=u \in \{1,\dots,U\}$ to index users and $i \in \{1,\dots,I\}"> to index items. In this work, we consider learning with implicit feedback. The user-by-item interaction matrix is the click matrix <img src="https://render.githubusercontent.com/render/math?math={X} \in \mathbb{N}^{U\times I}">. The lower case <img src="https://render.githubusercontent.com/render/math?math={x}_u =[X_{u1},\dots,X_{uI}]^\top \in \mathbb{N}^I"> is a bag-of-words vector with the number of clicks for each item from user u. We binarize the click matrix. It is straightforward to extend it to general count data.

__Generative process__: For each user <img src="https://render.githubusercontent.com/render/math?math=u">, the model starts by sampling a <img src="https://render.githubusercontent.com/render/math?math=K">-dimensional latent representation <img src="https://render.githubusercontent.com/render/math?math={z}_u"> from a standard Gaussian prior. The latent representation <img src="https://render.githubusercontent.com/render/math?math={z}_u"> is transformed via a non-linear function <img src="https://render.githubusercontent.com/render/math?math=f_\theta (\cdot) \in \mathbb{R}^I"> to produce a probability distribution over <img src="https://render.githubusercontent.com/render/math?math=I"> items <img src="https://render.githubusercontent.com/render/math?math=\pi (\mathbf{z}_u)"> from which the click history <img src="https://render.githubusercontent.com/render/math?math={x}_u"> is assumed to have been drawn:

<img align='center' src="https://render.githubusercontent.com/render/math?math={z}_u \sim \mathcal{N}(0, \mathbf{I}_K),  \pi(\mathbf{z}_u) \propto \exp\{f_\theta (\mathbf{z}_u\}">
<img align='center' src="https://render.githubusercontent.com/render/math?math={x}_u \sim \mathrm{Mult}(N_u, \pi(\mathbf{z}_u))">

The objective for Multi-DAE for a single user <img src="https://render.githubusercontent.com/render/math?math=u"> is:\
<img align='center' src="https://render.githubusercontent.com/render/math?math={L}_u(\theta, \phi) = \log p_\theta(\mathbf{x}_u | g_\phi(\mathbf{x}_u))"> \
where <img src="https://render.githubusercontent.com/render/math?math=g_\phi(\cdot)"> is the non-linear "encoder" function.


### Multi VAE

The objective of Multi-VAE^{PR} (evidence lower-bound, or ELBO) for a single user <img src="https://render.githubusercontent.com/render/math?math=u"> is:\
<img src="https://render.githubusercontent.com/render/math?math={L}_u(\theta, \phi) = \mathbb{E}_{q_\phi(z_u | x_u)}[\log p_\theta(x_u | z_u)] - \beta \cdot KL(q_\phi(z_u | x_u) \| p(z_u))">\
where <img src="https://render.githubusercontent.com/render/math?math=q_\phi"> is the approximating variational distribution (inference model). <img src="https://render.githubusercontent.com/render/math?math=beta"> is the additional annealing parameter that we control. The objective of the entire dataset is the average over all the users. It can be trained almost the same as Multi-DAE, thanks to reparametrization trick. 
    
    
## Evaluate

<img align='right' src='https://raw.githubusercontent.com/greyhub/RecommendationSystem/main/nDCG%40100.png' width='500"'>

      Test NDCG@100  = 0.43887 (0.00208)
      Test Recall@20 = 0.39996 (0.00261)
      Test Recall@50 = 0.53779 (0.00277)


## Implement

You can run directly with Google Colab 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greyhub/RecommendationSystem/blob/main/vae_cf.ipynb)


## References

This source code follows the paper "[Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)"  by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018.
    
-------------------------------------------------------------------------------------

