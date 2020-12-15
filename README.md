# Recommendation System

## Topic: Variational Autoencoders for Collaborative Filtering
> Team: Dang NM, Quan VD, Tuan BM, Luong NV

-------------------------------------------------------------------------------------

## Data

  + Main:
      + [MovieLens-25M](http://files.grouplens.org/datasets/movielens/ml-25m.zip)
  + Others: 
      + [MovieLens-20M](http://files.grouplens.org/datasets/movielens/ml-20m.zip)
      + [MovieLens-10M](http://files.grouplens.org/datasets/movielens/ml-10m.zip)


## Methods

  + Multi DAE
  + Multi VAE
    
    
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
