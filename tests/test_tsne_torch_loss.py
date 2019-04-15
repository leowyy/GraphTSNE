import unittest
from sklearn.manifold import TSNE
import numpy as np
import torch
from core.tsne_torch_loss import tsne_torch_loss, compute_joint_probabilities


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor

class TestTorchLoss(unittest.TestCase):
    def test_loss_equal(self):
        metrics = ['euclidean', 'cosine']
        for metric in metrics:
            n_train = 500
            n_dim = 100
            X = np.random.rand(n_train, n_dim)

            embedder = TSNE(n_components=2, method='exact', metric=metric, perplexity=30)
            X_emb = embedder.fit_transform(X)
            loss_sklearn = embedder.kl_divergence_

            P = compute_joint_probabilities(X, perplexity=30, metric=metric, method='approx')
            activations = torch.from_numpy(X_emb).type(dtypeFloat)
            loss_torch = tsne_torch_loss(P, activations)

            print(loss_sklearn)
            print(loss_torch)
            np.testing.assert_almost_equal(loss_sklearn, loss_torch, decimal=2)

if __name__ == '__main__':
    unittest.main()
