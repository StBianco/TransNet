import math

import numpy as np
import pytest
import torch

import TransNet_train as tt
from graph import graph

CPU = torch.device("cpu")
# 4 stations on a 0.3-side square: every pair is within 0.6 of each other.
LATS = np.array([0.0, 0.0, 0.3, 0.3])
LONS = np.array([0.0, 0.3, 0.0, 0.3])
N_STATIONS = 4
N_FEATURES = 43
SEQ = 24


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = tt.TransNet(N_FEATURES, N_STATIONS, SEQ, 72, L=1, state_dim=0, hist_dim=0).to(CPU)
    m.initialize_graph_processor(LATS, LONS)
    m.eval()
    return m


def _graph_inputs(model, batch=1, unit_weights=False):
    wind = torch.rand(batch, 2, N_STATIONS)
    edge_index, w, _ = model.graph_processor.construct_graph(wind)
    if unit_weights:
        w = torch.ones_like(w)
    G = graph(edge_index[:, 0], edge_index[:, 1], N_STATIONS, w, device=CPU)
    return edge_index, w, G


def _x(batch=1):
    torch.manual_seed(1)
    return torch.randn(batch, SEQ, N_STATIONS, N_FEATURES)


def test_setup_and_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(tt.dist, "init_process_group", lambda *a, **k: calls.append(("init", a, k)))
    monkeypatch.setattr(tt.dist, "destroy_process_group", lambda: calls.append(("destroy",)))
    tt.setup(rank=1, world_size=2)
    tt.cleanup()
    assert calls[0] == ("init", ("nccl",), {"rank": 1, "world_size": 2})
    assert calls[1] == ("destroy",)
    assert tt.os.environ["MASTER_ADDR"] == "localhost"


class TestDirectionalWeights:
    def test_forward_normalized(self):
        torch.manual_seed(0)
        gp = tt.GraphProcessor(LATS, LONS)
        edge_index = gp.edge_index.unsqueeze(0).repeat(2, 1, 1)
        dw = tt.DirectionalWeights(feature_dim=5, hidden_dim=8)
        Vij, Vji = dw(torch.randn(2, N_STATIONS, 5), edge_index, N_STATIONS)

        E = edge_index.shape[-1]
        assert Vij.shape == Vji.shape == (2, E)
        assert (Vij >= 0).all() and (Vji >= 0).all()
        src, dst = edge_index[0]
        for n in range(N_STATIONS):
            torch.testing.assert_close(Vij[:, src == n].sum(-1), torch.ones(2))
            torch.testing.assert_close(Vji[:, dst == n].sum(-1), torch.ones(2))


class TestGraphProcessor:
    def test_init_edges(self):
        gp = tt.GraphProcessor(LATS, LONS)
        src, dst = gp.edge_index
        assert gp.node_num == N_STATIONS
        assert (src != dst).all()  # no self loops
        assert gp.edge_index.shape == (2, N_STATIONS * (N_STATIONS - 1))  # fully connected
        assert gp.edge_attr.shape == gp.edge_attr_norm.shape == (gp.edge_index.shape[1], 2)

        coords = np.stack([LONS, LATS], axis=1)
        d = coords[dst] - coords[src]
        np.testing.assert_allclose(gp.city_distances, np.hypot(d[:, 0], d[:, 1]), rtol=1e-6)
        np.testing.assert_allclose(gp.city_directions, np.arctan2(d[:, 1], d[:, 0]), rtol=1e-6)

    def test_init_respects_dist_thres(self):
        # Neighbouring stations are 0.3 apart, so a 0.2 threshold leaves no edges.
        gp = tt.GraphProcessor(LATS, LONS, dist_thres=0.2)
        assert gp.edge_index.shape[1] == 0

    def test_construct_graph_wind_weights(self):
        gp = tt.GraphProcessor(LATS, LONS)
        E = gp.edge_index.shape[1]
        speed = 2.0
        # Wind for every source blowing exactly along edge 0 (direction in radians).
        wind = torch.zeros(3, 2, N_STATIONS)
        wind[:, 0] = speed
        wind[:, 1] = gp.city_directions[0]

        edge_index, w, feats = gp.construct_graph(wind)
        assert edge_index.shape == (3, 2, E)
        assert w.shape == (3, E)
        assert feats.shape == (3, E, 3)
        assert (w >= 0).all()
        torch.testing.assert_close(w[:, 0], torch.full((3,), speed / gp.city_distances[0].item()))
        torch.testing.assert_close(feats[..., -1], w)


class TestTransNet:
    def test_to_captures_device(self, model):
        assert model.device == CPU

    def test_get_K_clamps(self, model):
        out = model.get_K(torch.tensor([-1.0, 0.5, 2.0]))
        torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 1.0]))

    def test_diffusion_params_match_state_channels(self, model):
        # K is indexed per channel of the state in compute_diffusion_term.
        assert model.diffusion_params.numel() == model.state_latent_size

    def test_recover_wind_speed(self, model):
        u, v = torch.tensor([3.0, 0.0]), torch.tensor([4.0, -2.0])
        speed, _ = model.recover_wind_parameters(u, v)
        torch.testing.assert_close(speed, torch.tensor([5.0, 2.0]))

    @pytest.mark.parametrize("use_cos_sin", [False, True])
    def test_recover_wind_direction_in_0_360(self, model, use_cos_sin):
        angles = torch.linspace(-math.pi, math.pi, 17)
        u, v = torch.cos(angles), torch.sin(angles)
        args = (u, v, torch.cos(angles), torch.sin(angles)) if use_cos_sin else (u, v)
        _, direction = model.recover_wind_parameters(*args)
        assert (direction >= 0).all() and (direction < 360).all()

    def test_recover_wind_direction_units_match_graph_processor(self, model):
        # forward() feeds recover_wind_parameters' direction into construct_graph,
        # which compares it against city directions in radians. A wind blowing
        # exactly along an edge must give weight speed / distance.
        gp = model.graph_processor
        e = int(gp.city_directions.abs().argmax())  # a non-zero direction, e.g. pi
        ang = gp.city_directions[e]
        speed, direction = model.recover_wind_parameters(
            torch.cos(ang), torch.sin(ang), torch.cos(ang), torch.sin(ang))
        wind = torch.zeros(1, 2, N_STATIONS)
        wind[:, 0], wind[:, 1] = speed, direction
        _, w, _ = gp.construct_graph(wind)
        torch.testing.assert_close(w[0, e], speed / gp.city_distances[e])

    def test_advection_moves_mass_between_nodes(self, model):
        edge_index, w, G = _graph_inputs(model, unit_weights=True)
        C = model.hist_latent_size
        u_state = torch.zeros(1, C, N_STATIONS)
        u_state[:, :, 0] = 1.0  # all mass on station 0
        u_hist = torch.randn(1, C, N_STATIONS)
        out, V = model.compute_advection_term(u_state, u_hist, edge_index, w, G, h=0.5)
        assert out.shape == u_state.shape
        assert V.shape == w.shape
        # Station 0 is connected to every other station; something must flow out.
        assert out[:, :, 1:].abs().sum() > 0

    def test_diffusion_zero_K_is_identity(self, model):
        edge_index, w, G = _graph_inputs(model)
        u = torch.randn(1, 5, N_STATIONS)
        out = model.compute_diffusion_term(u, edge_index, w, G, torch.zeros(5), h=0.5)
        torch.testing.assert_close(out, u, atol=1e-5, rtol=1e-5)

    def test_diffusion_smooths(self, model):
        edge_index, w, G = _graph_inputs(model, unit_weights=True)
        u = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        out = model.compute_diffusion_term(u, edge_index, w, G, torch.ones(1), h=1.0)
        assert out.shape == u.shape
        assert out[0, 0].std() < u[0, 0].std()

    def test_reaction_shape_and_zero_step(self, model):
        S = model.state_latent_size
        state = torch.randn(1, N_STATIONS, S)
        hist = torch.randn(1, N_STATIONS, model.pollutant_size * SEQ)
        met = torch.randn(1, N_STATIONS, model.met_size)
        R = model.reaction_networks
        nets = (R["R1"], R["R2"], R["R3"], R["R4"])
        out = model.compute_reaction_term(state, hist, None, met, None, *nets, h=0.1)
        assert out.shape == state.shape
        # h = 0 leaves the state unchanged.
        out0 = model.compute_reaction_term(state, hist, None, met, None, *nets, h=0.0)
        torch.testing.assert_close(out0, state)

    def test_forward_shape(self, model):
        with torch.no_grad():
            out = model(_x())
        assert out.shape == (1, 72, N_STATIONS)
        assert torch.isfinite(out).all()

    def test_forward_propagates_state_across_layers(self, model, monkeypatch):
        model.L = 2
        adv_inputs, react_outputs = [], []
        orig_adv, orig_react = model.compute_advection_term, model.compute_reaction_term

        def adv(u_state, *a, **k):
            adv_inputs.append(u_state.clone())
            return orig_adv(u_state, *a, **k)

        def react(*a, **k):
            out = orig_react(*a, **k)
            react_outputs.append(out.clone())
            return out

        monkeypatch.setattr(model, "compute_advection_term", adv)
        monkeypatch.setattr(model, "compute_reaction_term", react)
        with torch.no_grad():
            model(_x())
        assert len(adv_inputs) == 2
        # Layer 2 must start from layer 1's output (in channel-first layout).
        torch.testing.assert_close(adv_inputs[1], react_outputs[0].transpose(1, 2))


class TestCustomLoss:
    def test_forward_perfect_prediction(self):
        target = torch.rand(2, 72, N_STATIONS) + 1
        mae, neg_ioa = tt.custom_loss()(target.clone(), target)
        torch.testing.assert_close(mae, torch.tensor(0.0))
        torch.testing.assert_close(neg_ioa, torch.tensor(-1.0))

    def test_forward_values(self):
        torch.manual_seed(0)
        pred, target = torch.rand(2, 72, N_STATIONS), torch.rand(2, 72, N_STATIONS)
        mae, neg_ioa = tt.custom_loss()(pred, target)
        torch.testing.assert_close(mae, (pred - target).abs().mean())
        assert -1.0 <= neg_ioa <= 0.0
