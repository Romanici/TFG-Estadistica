import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

#Ya no funciona
def acc(y_true, y_pred):
    from coclust.evaluation.external import accuracy
    acc = accuracy(y_true, y_pred)
 
    return(acc)

#from coclust.evaluation.external import accuracy

#acc = accuracy

"""
usage: DCEC.py          
               usage: DCEC.py [-h] [--n_clusters N_CLUSTERS] [--batch_size BATCH_SIZE]
               [--maxiter MAXITER] [--gamma GAMMA]
               [--update_interval UPDATE_INTERVAL] [--tol TOL]
               [--cae_weights CAE_WEIGHTS] [--save_dir SAVE_DIR]
               {mnist,usps,mnist-test,fashion}

train

positional arguments:
  {mnist,usps,mnist-test,fashion}

optional arguments:
  -h, --help            show this help message and exit
  --n_clusters N_CLUSTERS
  --batch_size BATCH_SIZE
  --maxiter MAXITER
  --gamma GAMMA         coefficient of clustering loss
  --update_interval UPDATE_INTERVAL
  --tol TOL
  --cae_weights CAE_WEIGHTS
                        This argument must be given
  --save_dir SAVE_DIR
"""               
# python dcec.py mnist-test --cae_weights "results/temp/pretrain_test_mnist_0.h5"

