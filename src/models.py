"""
models.py - Community detection models.

DISCLAIMER: This is a SIMPLIFIED re-implementation of DMRA-NMF for a
university course project. Key simplifications:
  - Undirected graphs only (symmetric factorization).
  - Triangle motif only (no higher-order motifs).
  - Trained with Adam optimizer (not multiplicative update rules).
  - Non-negativity enforced via softplus activation (not projection).
  - Number of communities k is assumed known.

This is NOT an exact reproduction of the original paper. We call our
simplified version "DMRA-lite" to be clear about this distinction.

Models implemented:
  1. SymNMF       : A ≈ H H^T  (single-layer symmetric NMF)
  2. DMRALite     : M ≈ Ψ W Ψ^T where Ψ = H1 H2 ... Hp (deep multi-layer)
                    + graph-Laplacian penalty + diagonal-dominance penalty
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.motif import triangle_motif_matrix, mix_matrices


# ---------------------------------------------------------------------------
# Helper: Graph Laplacian
# ---------------------------------------------------------------------------
def compute_laplacian(M: np.ndarray) -> np.ndarray:
    """Compute the combinatorial graph Laplacian: L = D - M."""
    D = np.diag(M.sum(axis=1))
    return (D - M).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. SymNMF (baseline, single-layer)
# ---------------------------------------------------------------------------
class SymNMF:
    """Single-layer Symmetric NMF: A ≈ H H^T.

    Uses multiplicative updates (Lee & Seung) for simplicity.
    """
    def __init__(self, k: int, max_iter: int = 200, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, A: np.ndarray, verbose: bool = False):
        """Fit SymNMF and return community labels."""
        n = A.shape[0]
        rng = np.random.RandomState(self.seed)

        # Initialize H randomly
        H = rng.rand(n, self.k).astype(np.float32) + 0.01

        losses = []
        for it in range(self.max_iter):
            # Multiplicative update for symmetric NMF: A ≈ H H^T
            HHt = H @ H.T
            numerator = A @ H
            denominator = HHt @ H + 1e-10
            H = H * (numerator / denominator)
            # Clamp to avoid division-by-zero cascades that can make the
            # multiplicative update overflow on the final iterations.
            H = np.clip(np.nan_to_num(H, nan=0.01, posinf=1e6, neginf=0.01), 1e-6, 1e6)

            if verbose and (it % 50 == 0 or it == self.max_iter - 1):
                loss = np.linalg.norm(A - H @ H.T, 'fro') ** 2
                losses.append(loss)
                print(f"    SymNMF iter {it}: loss = {loss:.2f}")

        self.H = H
        self.losses = losses
        labels = np.argmax(H, axis=1)
        return labels


# ---------------------------------------------------------------------------
# 2. DMRALite (main model, PyTorch)
# ---------------------------------------------------------------------------
class DMRALiteModel(nn.Module):
    """Deep Motif-Regularized NMF (simplified version).

    Factorization: M ≈ Ψ W Ψ^T
    where Ψ = H_1 @ H_2 @ ... @ H_p  (p layers, all non-negative)
    W is a k×k non-negative matrix (should be nearly diagonal for good clusters)

    Loss = ||M - Ψ W Ψ^T||_F^2
         + lambda * tr(Ψ^T L Ψ)        (graph Laplacian smoothness)
         + mu * ||W ∘ (1 - I)||_F^2     (penalize off-diagonal entries of W)
    """
    def __init__(self, layer_dims: list, lam: float = 1.0, mu: float = 1.0):
        """
        Args:
            layer_dims: list of dimensions, e.g. [n, 64, 32, k].
                        First element is n (number of nodes), last is k.
            lam: weight for Laplacian regularization.
            mu: weight for diagonal-dominance penalty on W.
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.lam = lam
        self.mu = mu
        self.p = len(layer_dims) - 1  # number of layers

        # Layer factors H_1, ..., H_p stored as raw parameters.
        # We use softplus to ensure non-negativity.
        #
        # Scale-aware initialisation is essential: if every factor starts with
        # softplus(0.1 * randn) ~ 0.69, their product is off by orders of
        # magnitude (e.g. 64 * 0.69^2 ~ 30 for a 2-layer model), the initial
        # loss is ~1e9, and training spends all its iterations merely shrinking
        # the factors instead of separating communities. We set each factor to
        # a value v such that the product Psi starts near PSI_TARGET.
        psi_target = 0.5
        mid_prod = float(np.prod(layer_dims[1:-1])) if self.p > 1 else 1.0
        v = float(np.clip((psi_target / mid_prod) ** (1.0 / self.p), 0.01, 0.5))
        raw_v = float(np.log(np.expm1(v)))
        self.H_raw = nn.ParameterList()
        for i in range(self.p):
            d_in = layer_dims[i]
            d_out = layer_dims[i + 1]
            param = nn.Parameter(
                torch.full((d_in, d_out), raw_v) + torch.randn(d_in, d_out) * 0.25
            )
            self.H_raw.append(param)

        # W: k×k matrix (last dimension).
        # CRITICAL: initialise W close to the identity. If W starts with large
        # off-diagonal entries, the factorisation becomes invariant to mixing
        # the columns of Psi and argmax(Psi) carries no community signal.
        # softplus^{-1}(y) = log(expm1(y)) gives a softplus-parameter that is
        # exactly y, so W starts as 1.0 on the diagonal and 0.01 off it.
        k = layer_dims[-1]
        w0 = np.full((k, k), 0.01, dtype=np.float32)
        np.fill_diagonal(w0, 1.0)
        w_raw0 = np.log(np.expm1(w0))
        self.W_raw = nn.Parameter(torch.from_numpy(w_raw0))

    def _get_nonneg(self, raw):
        """Apply softplus to ensure non-negativity."""
        return F.softplus(raw)

    def forward(self):
        """Compute Ψ and W (non-negative)."""
        # Ψ = H_1 @ H_2 @ ... @ H_p
        Psi = self._get_nonneg(self.H_raw[0])
        for i in range(1, self.p):
            Psi = Psi @ self._get_nonneg(self.H_raw[i])
        W = self._get_nonneg(self.W_raw)
        return Psi, W

    def compute_loss(self, M_torch, L_torch):
        """Compute the full objective loss."""
        Psi, W = self.forward()

        # Reconstruction: ||M - Ψ W Ψ^T||_F^2
        recon = Psi @ W @ Psi.T
        loss_recon = torch.norm(M_torch - recon, p='fro') ** 2

        # Laplacian: tr(Ψ^T L Ψ)
        loss_lap = torch.trace(Psi.T @ L_torch @ Psi)

        # Diagonal penalty: ||W ∘ (1 - I)||_F^2
        k = W.shape[0]
        mask = 1.0 - torch.eye(k, device=W.device)
        loss_diag = torch.norm(W * mask, p='fro') ** 2

        total = loss_recon + self.lam * loss_lap + self.mu * loss_diag
        return total, loss_recon.item(), loss_lap.item(), loss_diag.item()


def _nonneg_embedding(M, k, seed=42):
    """Return a non-negative node embedding E of shape (n, k) for a warm start.

    Tries sklearn NMF first (best community signal), then a spectral fallback
    based on the leading eigenvectors of M, then a deterministic random matrix.
    The embedding is scaled to [0, 1] so that the initial Psi has a sane scale.
    """
    n = M.shape[0]
    kk = min(k, n)
    E = None
    try:
        from sklearn.decomposition import NMF
        E = NMF(n_components=kk, init='random', random_state=seed,
                max_iter=300).fit_transform(np.maximum(M, 0))
    except Exception:
        pass
    if E is None:
        try:
            _, vecs = np.linalg.eigh(M.astype(np.float64))
            E = np.abs(vecs[:, -kk:])
        except Exception:
            E = np.random.RandomState(seed).rand(n, kk)
    if E.shape[1] < k:
        E = np.hstack([E, np.zeros((n, k - E.shape[1]))])
    E = np.maximum(E, 0)
    scale = E.max()
    if scale > 0:
        E = E / scale
    return E.astype(np.float32)


def _apply_structured_init(model, E, layer_dims):
    """Initialise H_1 ... H_p so that Psi starts close to the NMF embedding E.

    The first factor stores E in its first k columns; every later factor has
    an identity block in its first k rows/columns and a small positive
    background elsewhere. Their product therefore starts as ~0.5 * E, which
    already encodes the community structure. This is what prevents the deep
    factorisation from collapsing into a rank-1 (all columns identical)
    solution that reconstructs the graph but yields random cluster labels.
    """
    k = layer_dims[-1]
    background = 0.01
    with torch.no_grad():
        for i, param in enumerate(model.H_raw):
            d_in, d_out = layer_dims[i], layer_dims[i + 1]
            target = np.full((d_in, d_out), background, dtype=np.float32)
            if i == 0:
                m = min(k, d_out)
                target[:, :m] = E[:, :m] * 0.5
            else:
                m = min(k, d_in, d_out)
                if m > 0:
                    target[:m, :m] = np.eye(m, dtype=np.float32)
            # Keep targets strictly positive: softplus^{-1}(0) = -inf, and
            # -inf parameters produce NaN gradients during training.
            raw = np.log(np.expm1(np.maximum(target, 1e-4)))
            param.copy_(torch.from_numpy(raw.astype(np.float32)))


def _choose_layer_dims(n: int, k: int, depth: int = 2) -> list:
    """Choose reasonable intermediate layer dimensions.
    Returns e.g. [n, 64, k] for depth=2 on a medium graph."""
    if depth == 1:
        return [n, k]
    elif depth == 2:
        mid = max(k * 2, min(64, n // 2))
        return [n, mid, k]
    elif depth >= 3:
        mid1 = max(k * 4, min(128, n // 2))
        mid2 = max(k * 2, min(64, n // 4))
        dims = [n, mid1, mid2, k]
        return dims[:depth + 1]  # trim if depth < 3
    return [n, k]


class DMRALite:
    """High-level wrapper for the DMRA-lite deep NMF model.

    Usage:
        model = DMRALite(k=7, alpha=0.5)
        labels = model.fit(A, verbose=True)
    """
    def __init__(self, k: int, alpha: float = 0.5,
                 lam: float = 1.0, mu: float = 1.0,
                 depth: int = 2, lr: float = 0.005,
                 max_iter: int = 300, seed: int = 42):
        """
        Args:
            k: number of communities.
            alpha: motif mixing weight (0 = no motif, 1 = motif only).
            lam: Laplacian regularization weight.
            mu: diagonal-dominance penalty weight.
            depth: number of factorization layers (1, 2, or 3).
            lr: learning rate for Adam.
            max_iter: number of training iterations.
            seed: random seed.
        """
        self.k = k
        self.alpha = alpha
        self.lam = lam
        self.mu = mu
        self.depth = depth
        self.lr = lr
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, A: np.ndarray, verbose: bool = False):
        """Fit the model on adjacency matrix A and return community labels.

        Args:
            A: (n, n) symmetric adjacency matrix.
            verbose: if True, print loss every 50 iterations.

        Returns:
            labels: (n,) int array of community assignments.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n = A.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute motif matrix and mix
        B = triangle_motif_matrix(A)
        M = mix_matrices(A, B, self.alpha)

        # Compute Laplacian of M
        L = compute_laplacian(M)

        # Choose layer dimensions
        layer_dims = _choose_layer_dims(n, self.k, self.depth)

        # Initialize model
        model = DMRALiteModel(layer_dims, lam=self.lam, mu=self.mu).to(device)

        # Structured warm start: initialise the factor chain so that Psi is
        # already close to a non-negative community embedding of M.
        E = _nonneg_embedding(M, self.k, seed=self.seed)
        _apply_structured_init(model, E, layer_dims)

        M_torch = torch.from_numpy(M).to(device)
        L_torch = torch.from_numpy(L).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        losses = []
        for it in range(self.max_iter):
            optimizer.zero_grad()
            total_loss, l_recon, l_lap, l_diag = model.compute_loss(M_torch, L_torch)
            total_loss.backward()
            optimizer.step()

            if verbose and (it % 50 == 0 or it == self.max_iter - 1):
                print(f"    DMRA-lite iter {it}: total={total_loss.item():.2f} "
                      f"recon={l_recon:.2f} lap={l_lap:.2f} diag={l_diag:.2f}")
            losses.append(total_loss.item())

        # Check if loss decreased
        if len(losses) >= 2:
            if losses[-1] > losses[0] * 1.1:
                if verbose:
                    print(f"  WARNING: loss increased! Start={losses[0]:.2f} End={losses[-1]:.2f}")
            elif verbose:
                print(f"  Loss decreased: {losses[0]:.2f} -> {losses[-1]:.2f} (good)")

        # Extract community labels from Ψ
        model.eval()
        with torch.no_grad():
            Psi, W = model.forward()
            Psi_np = Psi.cpu().numpy()

        labels = np.argmax(Psi_np, axis=1)

        self.losses = losses
        self.Psi = Psi_np
        self.model = model
        return labels
