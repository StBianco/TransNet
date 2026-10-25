import pytest
import torch

from graph import graph

CPU = torch.device("cpu")
B, C, N = 2, 3, 3
# Path graph 0 - 1 - 2 with both edge directions, same edges for every batch.
I_IND = torch.tensor([0, 1, 1, 2])
J_IND = torch.tensor([1, 0, 2, 1])
E = I_IND.numel()


@pytest.fixture
def W():
    return torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 1.0, 1.0]])


@pytest.fixture
def G(W):
    return graph(I_IND.repeat(B, 1), J_IND.repeat(B, 1), N, W, CPU)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(B, C, N)


def _dense_sym_laplacian(A):
    d = A.sum(1).pow(-0.5)
    return torch.eye(A.shape[0]) - d[:, None] * A * d[None, :]


class TestGraph:
    def test_init_stores_tensors(self, G, W):
        assert G.iInd.dtype == torch.long and G.jInd.dtype == torch.long
        assert G.iInd.shape == (B, E)
        assert G.nnodes == N
        assert torch.equal(G.W, W)
        assert G.pos is None and G.faces is None

    def test_nodeGrad_values(self, G, W, x):
        g = G.nodeGrad(x)
        assert g.shape == (B, C, E)
        expected = W.unsqueeze(1) * (x[:, :, J_IND] - x[:, :, I_IND])
        torch.testing.assert_close(g, expected)

    def test_nodeGrad_custom_weights(self, G, W, x):
        W2 = torch.full_like(W, 2.0)
        torch.testing.assert_close(G.nodeGrad(x, W2), 2.0 * G.nodeGrad(x))
        torch.testing.assert_close(G.nodeGrad(x, W2, replaceW=True), G.nodeGrad(x, W2 / W))

    def test_edgeDiv_is_negative_adjoint_of_nodeGrad(self, G, x):
        # <grad x, g> == -<x, div g> for the weighted operators.
        g = torch.randn(B, C, E)
        lhs = (G.nodeGrad(x) * g).sum()
        rhs = -(x * G.edgeDiv(g)).sum()
        torch.testing.assert_close(lhs, rhs)

    def test_nodeLap_of_constant_is_zero(self, G):
        x = torch.ones(B, C, N)
        torch.testing.assert_close(G.nodeLap(x), torch.zeros(B, C, N))

    def test_edgeLength_is_norm_of_gradient(self, G, x):
        L = G.edgeLength(x)
        assert L.shape == (B, E)
        torch.testing.assert_close(L, G.nodeGrad(x).norm(dim=1))

    def test_nodeAve_explicit_weights(self, G, W, x):
        g = G.nodeAve(x, W)
        expected = W.unsqueeze(1) * (x[:, :, I_IND] + x[:, :, J_IND]) / 2
        torch.testing.assert_close(g, expected)

    def test_nodeAve_default_weights(self, G, W, x):
        # W is documented as optional (default None).
        torch.testing.assert_close(G.nodeAve(x), G.nodeAve(x, W))

    def test_neighborNode_values(self, G, W, x):
        g = G.neighborNode(x)
        assert g.shape == (B, C, E)
        torch.testing.assert_close(g, W.unsqueeze(1) * x[:, :, J_IND])

    # The following operators should accept batched (B, num_edges) edge indices.

    def test_nodeProd_batched_shape(self, G, x):
        assert G.nodeProd(x).shape == (B, C, E)

    def test_neighborEdge_batched_shape(self, G):
        assert G.neighborEdge(torch.randn(B, C, E)).shape == (B, C, N)

    @pytest.mark.parametrize("method", ["max", "ave"])
    def test_edgeAve_batched_shape(self, G, method):
        assert G.edgeAve(torch.randn(B, C, E), method=method).shape == (B, C, N)

    def test_dirichletEnergy_batched_shape(self, G, x):
        assert G.dirichletEnergy(x, torch.ones(N)).shape == (B, C, E)

    def test_normalized_laplacian_symmetric_weights(self):
        # Unbatched usage, as in TransNet.compute_diffusion_term.
        W = torch.tensor([2.0, 2.0, 3.0, 3.0])
        L = graph(I_IND, J_IND, N, W, CPU).get_normalized_laplacian()
        A = torch.tensor([[0.0, 2.0, 0.0], [2.0, 0.0, 3.0], [0.0, 3.0, 0.0]])
        torch.testing.assert_close(L, _dense_sym_laplacian(A))

    def test_normalized_laplacian_is_symmetric_for_directed_weights(self):
        # Wind-based weights differ per direction; the Laplacian is documented
        # as using a symmetric adjacency.
        W = torch.tensor([1.0, 0.0, 5.0, 0.0])
        L = graph(I_IND, J_IND, N, W, CPU).get_normalized_laplacian()
        torch.testing.assert_close(L, L.T)
