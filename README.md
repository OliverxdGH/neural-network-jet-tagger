# Neural Network Jet Tagger (Cuts vs MLP)

Cut-based and ML-based jet-tagging style analysis in Python.

This script provides lightweight collider-style data structures (`Particle`, `Event`), loads a simple CSV format into events, engineers jet–lepton features, then runs:
- A **cut-based baseline selection**
- A **supervised PyTorch MLP** trained on **MC truth labels** with **group-aware train/val/test splits**
- Application of the trained classifier to **real data** and comparison of **jet mass spectra**

This is intended as a compact exercise / study workflow, not a full experiment framework.

## Dataset format

CSV lines (comma-separated), one particle per line. Lines starting with `#` are ignored.

### MC file (with truth label)
Columns:
- event_id, pid, pt, eta, phi, e, m, truth_flag
- `truth_flag` should be `0` or `1`
- `truth_flag` is only meaningful for jets


### Cnventions used here

- Jets are identified by `|pid| == 90`
- Everything else is treated as “lepton-like”
- Per event, the analysis uses:
  - the **leading jet** (highest `pt` jet)
  - the **leading lepton** (highest `pt` non-jet)

## Selection (cut-based baseline)

An event passes the baseline selection if it has both a leading jet and a leading lepton, and:

- `jet_pt >= 250 GeV`
- `|jet_eta| <= 2.0`
- `lepton_pt >= 50 GeV`
- `dphi(jet, lepton) >= 2.4`

These are defined in the script as:
- `min_pt_j = 250.0`
- `min_pt_l = 50.0`
- `min_dphi = 2.4`
- `eta_j_max = 2.0`

## Features

For each event, features are built from the leading jet `j` and leading lepton `l`:

1. `jet_pt`
2. `jet_eta`
3. `jet_phi`
4. `lep_pt`
5. `lep_eta`
6. `lep_phi`
7. `dphi(j,l)` (wrapped to `[0, pi]`)
8. `dR(j,l)` in `(eta, phi)`
9. `pt_ratio = jet_pt / lep_pt`
10. `pt_asymmetry = (jet_pt - lep_pt) / (jet_pt + lep_pt)`

Implementation: `make_feats(j, l)`.

## What the script produces

### A) Data: mass spectra before/after cuts
- Reads `jets.csv`
- Plots the large-R jet mass distribution:
  - all selected events with a leading jet and lepton
  - events passing the baseline cuts
- Prints cut efficiency on data

### B) MC: baseline vs cuts (purity and efficiencies)
- Reads `pythia.csv`
- Computes:
  - baseline purity (no cuts)
  - purity after cuts: `S / (S + B)`
  - signal efficiency: `eps_S = S_after / S_before`
  - background efficiency: `eps_B = B_after / B_before`

### C) ML pipeline (MC training, evaluation, working point)
- Builds a supervised dataset on MC:
  - label: `y = truth_flag` for the leading jet
  - groups: `event_id` for group-aware splitting
- Splits MC with `GroupShuffleSplit` (by `event_id`):
  - train / validation / test
- Standardizes features using `StandardScaler` fit on train only
- Trains a small PyTorch MLP:
  - architecture: `10 -> 32 -> 16 -> 1`
  - loss: `BCEWithLogitsLoss` with `pos_weight` for class imbalance
  - optimizer: Adam
  - early stopping on validation loss
- Evaluates:
  - validation AUC
  - test AUC
  - ROC curve on test

#### Working point (threshold selection)
A threshold `t*` is chosen on the **MC test set** to maximize purity while enforcing a minimum signal efficiency:
- Constraint: `εS >= 0.30`
- Choose `t*` that maximizes **purity (precision)** among thresholds satisfying the constraint

The script also plots purity vs threshold and indicates `t*`.

### D) Apply trained NN to real data
- Applies the trained model to the data sample (`jets.csv`)
- Compares jet mass distributions:
  - all data vs NN selection
  - all data vs baseline cuts vs NN selection
- Prints MC test-set purity comparison:
  - baseline cuts purity
  - NN(t*) purity

## Requirements

- Python 3.13 (3.9+ likely fine)
- `numpy`
- `matplotlib`
- `scikit-learn`
- `torch` (PyTorch)

Install:
```bash
pip install numpy matplotlib scikit-learn torch
