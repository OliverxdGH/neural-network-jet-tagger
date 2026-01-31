import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

"""
Cut-based and ML-based jet tagging style analysis.

This file defines lightweight data structures (Particle, Event), I/O helpers to load
a CSV dataset into events, feature engineering utilities, and then runs:
  - A cut-based baseline selection
  - A supervised ML pipeline (PyTorch MLP) on MC with truth labels
  - Application of the trained classifier to real data

Notes
-----
- CSV format supported:
    * MC:   event_id, pid, pt, eta, phi, e, m, truth_flag
    * Data: event_id, pid, pt, eta, phi, e, m
- Jets are identified by PID = 90.
- "truth" is only meaningful for jets in MC samples.
"""

# Configure matplotlib settings for consistent plotting
plt.rcParams["figure.figsize"] = (5, 4)
plt.rcParams["axes.grid"] = True


# -----------------------------
# Basic data structures
# -----------------------------
class Particle:
    """
    Particle-level record for collider analysis.

    Attributes
    ----------
    event_id
        Collision event identifier this particle belongs to.
    pid
        Particle ID (PDG-like). Convention here: 90 indicates a jet object.
    pt, eta, phi
        Kinematic variables (pt in GeV, phi in radians).
    e, m
        Energy and invariant mass (GeV).
    truth
        For MC jets only: True if jet is signal-like (from a heavy particle decay).
    """

    def __init__(self, event_id: int, pid: int, pt: float, eta: float, phi: float, e: float, m: float, truth: bool = False) -> None:
        self.event_id = event_id
        self.pid = pid
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.e = e
        self.m = m
        self.truth = truth


class Event:
    """
    Container for all Particle objects in one collision event.

    Methods provide convenience selectors for jets, leptons, and leading objects.
    """

    def __init__(self, eid: int, particles: Optional[List[Particle]] = None) -> None:
        self.event_id = eid
        self.particles: List[Particle] = [] if particles is None else particles

    def add(self, p: Particle) -> None:
        """Add a particle to this event, asserting consistent event_id."""
        assert p.event_id == self.event_id
        self.particles.append(p)

    def jets(self) -> List[Particle]:
        """Return all jets in the event (PID convention: |pid| == 90)."""
        return [p for p in self.particles if abs(p.pid) == 90]

    def leptons(self) -> List[Particle]:
        """
        Return all non-jet objects in the event.

        Note: in this exercise, everything with |pid| != 90 is treated as a lepton-like object.
        """
        return [p for p in self.particles if abs(p.pid) != 90]

    def leading_jet(self) -> Optional[Particle]:
        """Return the jet with highest pt, or None if no jets exist."""
        js = self.jets()
        return max(js, key=lambda p: p.pt) if js else None

    def leading_lepton(self) -> Optional[Particle]:
        """Return the lepton with highest pt, or None if no leptons exist."""
        ls = self.leptons()
        return max(ls, key=lambda p: p.pt) if ls else None


# -----------------------------
# IO: load CSV (MC or data)
# -----------------------------
def load_events_csv(path: str) -> List[Event]:
    """
    Load a CSV file into a list of Event objects sorted by event_id.

    Supported formats
    -----------------
    - MC:   8 columns (event_id, pid, pt, eta, phi, e, m, truth_flag)
    - Data: 7 columns (event_id, pid, pt, eta, phi, e, m)

    Lines starting with '#' and empty lines are ignored.
    Malformed lines are skipped.
    """
    events: Dict[int, Event] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue

            toks = line.split(",")
            truth = False

            if len(toks) == 8:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s, truth_s = toks
                truth = bool(int(truth_s))
            elif len(toks) == 7:
                eid_s, pid_s, pt_s, eta_s, phi_s, e_s, m_s = toks
            else:
                continue

            eid = int(eid_s)
            pid = int(pid_s)
            pt = float(pt_s)
            eta = float(eta_s)
            phi = float(phi_s)
            e = float(e_s)
            m = float(m_s)

            if eid not in events:
                events[eid] = Event(eid)

            events[eid].add(Particle(eid, pid, pt, eta, phi, e, m, truth))

    return [events[k] for k in sorted(events)]


# -----------------------------
# Helpers
# -----------------------------
def dphi(a: float, b: float) -> float:
    """
    Minimal absolute difference in azimuthal angle phi, accounting for 2π periodicity.

    Returns a value in [0, π].
    """
    d = a - b
    return abs((d + math.pi) % (2.0 * math.pi) - math.pi)


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """
    Delta-R distance in (eta, phi) space: sqrt((Δeta)^2 + (Δphi)^2).
    """
    return math.hypot(eta1 - eta2, dphi(phi1, phi2))


def make_feats(j: Particle, l: Particle) -> List[float]:
    """
    Build a 10D feature vector from a jet and lepton.

    Features
    --------
    [jet_pt, jet_eta, jet_phi,
     lep_pt, lep_eta, lep_phi,
     dphi(j,l), dR(j,l), pt_ratio, pt_asymmetry]
    """
    dphi_jl = dphi(j.phi, l.phi)
    dR_jl = delta_r(j.eta, j.phi, l.eta, l.phi)

    pt_ratio = j.pt / l.pt if l.pt > 0.0 else 0.0
    denom = (j.pt + l.pt)
    pt_asym = (j.pt - l.pt) / denom if denom > 0.0 else 0.0

    return [j.pt, j.eta, j.phi, l.pt, l.eta, l.phi, dphi_jl, dR_jl, pt_ratio, pt_asym]


# -----------------------------
# Main analysis
# -----------------------------
if __name__ == "__main__":
    """
    Run the full analysis:
      A) Cut-based baseline
      B) ML training on MC with group-aware splitting
      C) Apply NN to real data and compare mass spectra
    """

    DATA_FILE = "jets.csv"
    MC_FILE = "pythia.csv"

    # Cut parameters
    min_pt_j = 250.0
    min_pt_l = 50.0
    min_dphi = 2.4
    eta_j_max = 2.0

    def pass_cuts(e: Event) -> bool:
        """
        Apply a simple cut-based selection on an event.

        Requirements:
          - Event has a leading jet and leading lepton
          - jet pt >= min_pt_j and |jet eta| <= eta_j_max
          - lepton pt >= min_pt_l
          - dphi(jet, lepton) >= min_dphi
        """
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            return False

        return (j.pt >= min_pt_j) and (l.pt >= min_pt_l) and (dphi(j.phi, l.phi) >= min_dphi) and (abs(j.eta) <= eta_j_max)

    # ---- Data: mass spectra before/after cuts
    data_events = load_events_csv(DATA_FILE)

    all_masses: List[float] = []
    sel_masses_cuts: List[float] = []
    seen = 0
    kept = 0

    for e in data_events:
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            continue

        seen += 1
        all_masses.append(j.m)

        if pass_cuts(e):
            kept += 1
            sel_masses_cuts.append(j.m)

    efficiency = 100.0 * kept / max(1, seen)
    print(f"Data (cuts): selected {kept}/{seen} = {efficiency:.1f}%.")

    bins = 40
    mass_range = (60.0, 140.0)

    plt.figure(figsize=(5.8, 4.2))
    plt.hist(all_masses, bins=bins, range=mass_range, density=True, histtype="step", label="All data", linewidth=1.5)
    plt.hist(sel_masses_cuts, bins=bins, range=mass_range, density=True, histtype="step", label="Cuts", linewidth=1.5)
    plt.xlabel("Large-R jet mass [GeV]")
    plt.ylabel("Density")
    plt.title("Data: jet mass before/after cuts")
    plt.legend()
    plt.tight_layout()

    # ---- MC: purity and efficiencies for the same cuts
    mc_events = load_events_csv(MC_FILE)

    S0 = 0
    B0 = 0
    for e in mc_events:
        j = e.leading_jet()
        if j is None:
            continue
        S0 += int(j.truth)
        B0 += int(not j.truth)

    S = 0
    B = 0
    for e in mc_events:
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            continue
        if pass_cuts(e):
            if j.truth:
                S += 1
            else:
                B += 1

    N_sel = S + B
    purity_mc = (S / N_sel) if N_sel > 0 else float("nan")
    eps_S = S / S0 if S0 > 0 else float("nan")
    eps_B = B / B0 if B0 > 0 else float("nan")
    purity0 = S0 / max(1, (S0 + B0))

    print(f"Baseline purity (no cuts) = {purity0:.3f}  with S0={S0}, B0={B0}")
    print(f"MC (cuts): purity S/(S+B) = {purity_mc:.3f}  with S={S}, B={B}, N={N_sel}  |  eps_S={eps_S:.3f}, eps_B={eps_B:.3f}")

    plt.show()

    # -----------------------------
    # B.1 Feature engineering on MC
    # -----------------------------
    X_list: List[List[float]] = []
    y_list: List[int] = []
    groups_list: List[int] = []
    ev_refs: List[Event] = []

    for e in mc_events:
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            continue

        X_list.append(make_feats(j, l))
        y_list.append(int(j.truth))
        groups_list.append(e.event_id)
        ev_refs.append(e)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups_list, dtype=np.int64)

    # -----------------------------
    # B.2 Split MC by event id
    # -----------------------------
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=42)
    train_idx, hold_idx = next(gss.split(X, y, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=43)
    val_rel, test_rel = next(gss2.split(X[hold_idx], y[hold_idx], groups=groups[hold_idx]))

    val_idx = hold_idx[val_rel]
    test_idx = hold_idx[test_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    print(f"Split sizes: train={len(X_train_s)}, val={len(X_val_s)}, test={len(X_test_s)}")

    # -----------------------------
    # B.3 PyTorch model
    # -----------------------------
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SmallMLP(nn.Module):
        """
        Small MLP binary classifier producing a single logit.

        Architecture:
          in_dim -> 32 -> 16 -> 1
        """

        def __init__(self, in_dim: int = 10, p_drop: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Dropout(p_drop),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass returning logits."""
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SmallMLP(in_dim=int(X_train_s.shape[1]), p_drop=0.1).to(device)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def make_loader(Xn: np.ndarray, yn: np.ndarray, batch: int = 1024, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader from numpy arrays.

        yn is converted to float32 column vector for BCEWithLogitsLoss.
        """
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xn),
            torch.from_numpy(yn.astype(np.float32)).unsqueeze(1),
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(X_train_s, y_train, batch=1024, shuffle=True)
    val_loader = make_loader(X_val_s, y_val, batch=4096, shuffle=False)
    test_loader = make_loader(X_test_s, y_test, batch=4096, shuffle=False)

    from sklearn.metrics import roc_auc_score, roc_curve

    def eval_loss_and_logits(loader: torch.utils.data.DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a loader.

        Returns
        -------
        avg_loss
            Mean loss over samples.
        logits
            Flattened logits for all samples.
        y_true
            Flattened integer labels for all samples.
        """
        model.eval()
        tot_loss = 0.0
        n_obs = 0
        all_logits: List[np.ndarray] = []
        all_y: List[np.ndarray] = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                tot_loss += float(loss.item()) * xb.size(0)
                n_obs += int(xb.size(0))

                all_logits.append(logits.detach().cpu().numpy().ravel())
                all_y.append(yb.detach().cpu().numpy().ravel())

        avg_loss = tot_loss / max(1, n_obs)
        logits_out = np.concatenate(all_logits) if all_logits else np.array([])
        y_out = np.concatenate(all_y).astype(int) if all_y else np.array([], dtype=int)
        return avg_loss, logits_out, y_out

    # -----------------------------
    # B.4 Training with early stopping
    # -----------------------------
    epochs = 400
    patience = 100
    min_delta = 1e-4

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    wait = 0

    train_losses: List[float] = []
    val_losses: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            seen_train += int(xb.size(0))

        train_loss = running_loss / max(1, seen_train)
        val_loss, _, _ = eval_loss_and_logits(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = (best_val_loss - val_loss) > min_delta
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            wait = 0
        else:
            wait += 1

        if ep % 5 == 0 or ep == 1:
            status = "(best)" if improved else ""
            print(f"Epoch {ep:03d}/{epochs}  TrainLoss={train_loss:.5f}  ValLoss={val_loss:.5f}  {status}")

        if wait > patience:
            print(f"Early stopping at epoch {ep}. Best epoch was {best_epoch} with ValLoss={best_val_loss:.5f}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    else:
        print("Warning: best_state is None, using final epoch weights.")

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    if best_epoch > 0:
        plt.axvline(best_epoch - 1, linestyle="--", linewidth=1, label=f"Best epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    _, val_logits, y_val_i = eval_loss_and_logits(val_loader)
    _, test_logits, y_test_i = eval_loss_and_logits(test_loader)

    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))

    val_auc = roc_auc_score(y_val_i, val_probs) if len(y_val_i) > 0 else float("nan")
    test_auc = roc_auc_score(y_test_i, test_probs) if len(y_test_i) > 0 else float("nan")

    print(f"Validation AUC = {val_auc:.3f}")
    print(f"Test AUC       = {test_auc:.3f}")
    print(f"Best epoch     = {best_epoch}  with ValLoss={best_val_loss:.5f}")

    # -----------------------------
    # B.5 Test ROC and threshold working point
    # -----------------------------
    from sklearn.metrics import roc_curve as sk_roc_curve, roc_auc_score as sk_roc_auc_score

    fpr_test, tpr_test, thr_test = sk_roc_curve(y_test_i.astype(int), test_probs)
    auc_test_check = sk_roc_auc_score(y_test_i.astype(int), test_probs)
    print(f"[B.5] Test ROC AUC = {auc_test_check:.3f}")

    plt.figure()
    plt.plot(fpr_test, tpr_test, label=f"NN (test AUC={auc_test_check:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random classifier")
    plt.xlabel("Background efficiency (FPR = εB)")
    plt.ylabel("Signal efficiency (TPR = εS)")
    plt.title("ROC on test")
    plt.legend()
    plt.tight_layout()
    plt.show()

    y_true = y_test_i.astype(int)
    scores = test_probs

    thr_grid = np.linspace(0.0, 1.0, 501)
    purities = np.zeros_like(thr_grid, dtype=float)
    tprs = np.zeros_like(thr_grid, dtype=float)
    fprs = np.zeros_like(thr_grid, dtype=float)

    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    prevalence = P / max(1, (P + N))

    for i, t in enumerate(thr_grid):
        y_hat = (scores >= t).astype(int)

        TP = int(np.sum((y_hat == 1) & (y_true == 1)))
        FP = int(np.sum((y_hat == 1) & (y_true == 0)))
        FN = int(np.sum((y_hat == 0) & (y_true == 1)))
        TN = int(np.sum((y_hat == 0) & (y_true == 0)))

        purities[i] = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        tprs[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fprs[i] = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    plt.figure()
    plt.plot(thr_grid, purities, label="Purity (precision)")
    plt.axhline(prevalence, linestyle="--", color="gray", label=f"Class prevalence = {prevalence:.3f}")
    plt.xlabel("Threshold t")
    plt.ylabel("Purity")
    plt.title("Purity vs threshold on test")
    plt.legend()
    plt.tight_layout()
    plt.show()

    mask = tprs >= 0.30
    if np.any(mask):
        idx_star = int(np.nanargmax(np.where(mask, purities, np.nan)))
        t_star = float(thr_grid[idx_star])
        purity_star = float(purities[idx_star])
        tpr_star = float(tprs[idx_star])
        fpr_star = float(fprs[idx_star])

        print(f"[B.5] Working point t* = {t_star:.4f}")
        print(f"[B.5] At t*: purity={purity_star:.3f}, εS={tpr_star:.3f}, εB={fpr_star:.3f}")
    else:
        t_star = 0.5
        print("[B.5] No threshold satisfies εS >= 0.30. Falling back to t*=0.5.")

    plt.figure()
    plt.plot(thr_grid, purities, label="Purity (precision)")
    plt.axhline(prevalence, linestyle="--", color="gray", label=f"Class prevalence = {prevalence:.3f}")
    plt.axvline(t_star, linestyle="--", color="red", label=f"t* = {t_star:.4f}")
    plt.xlabel("Threshold t")
    plt.ylabel("Purity")
    plt.title("Purity vs threshold on test with t*")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # C. Apply NN to real data
    # -----------------------------
    X_data_list: List[List[float]] = []
    masses_all: List[float] = []
    masses_cuts: List[float] = []

    for e in data_events:
        j = e.leading_jet()
        l = e.leading_lepton()
        if (j is None) or (l is None):
            continue

        X_data_list.append(make_feats(j, l))
        masses_all.append(j.m)
        if pass_cuts(e):
            masses_cuts.append(j.m)

    if len(X_data_list) > 0:
        X_data = np.array(X_data_list, dtype=np.float32)
        X_data_s = scaler.transform(X_data).astype(np.float32)

        with torch.no_grad():
            logits_data = model(torch.from_numpy(X_data_s).to(device)).cpu().numpy().ravel()
        scores_data = 1.0 / (1.0 + np.exp(-logits_data))

        masses_nn_t = [m for m, s in zip(masses_all, scores_data) if s >= t_star]

        plt.figure(figsize=(5.8, 4.2))
        plt.hist(masses_all, bins=40, range=(60, 140), density=True, histtype="step", label="All data", linewidth=1.5)
        plt.hist(masses_nn_t, bins=40, range=(60, 140), density=True, histtype="step", label=f"NN selection (t*={t_star:.3f})", linewidth=1.5)
        plt.xlabel("Large-R jet mass [GeV]")
        plt.ylabel("Density")
        plt.title("Data mass: before vs NN(t*)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5.8, 4.2))
        plt.hist(masses_all, bins=40, range=(60, 140), density=True, histtype="step", label="All data", linewidth=1.5)
        plt.hist(masses_cuts, bins=40, range=(60, 140), density=True, histtype="step", label="Baseline cuts", linewidth=1.5)
        plt.hist(masses_nn_t, bins=40, range=(60, 140), density=True, histtype="step", label=f"NN (t*={t_star:.3f})", linewidth=1.5)
        plt.xlabel("Large-R jet mass [GeV]")
        plt.ylabel("Density")
        plt.title("Data mass: all vs cuts vs NN(t*)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Compare purities on MC test set for baseline cuts vs NN(t*)
        S_cuts = 0
        B_cuts = 0
        for ridx in test_idx:
            ev = ev_refs[int(ridx)]
            j = ev.leading_jet()
            l = ev.leading_lepton()
            if (j is None) or (l is None):
                continue
            if pass_cuts(ev):
                if j.truth:
                    S_cuts += 1
                else:
                    B_cuts += 1
        purity_cuts = (S_cuts / (S_cuts + B_cuts)) if (S_cuts + B_cuts) > 0 else float("nan")

        y_sel_nn = (scores >= t_star).astype(int)
        S_nn = int(np.sum((y_sel_nn == 1) & (y_true == 1)))
        B_nn = int(np.sum((y_sel_nn == 1) & (y_true == 0)))
        purity_nn = (S_nn / (S_nn + B_nn)) if (S_nn + B_nn) > 0 else float("nan")

        print("C3 MC test purities:")
        print(f" - Baseline cuts purity: {purity_cuts:.3f}  (S={S_cuts}, B={B_cuts})")
        print(f" - NN(t*) purity:        {purity_nn:.3f}  (S={S_nn}, B={B_nn})")
    else:
        print("No selectable events found in data for C1-C3.")
