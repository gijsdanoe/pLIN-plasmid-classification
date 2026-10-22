#!/usr/bin/env python3
"""Head-to-head on pLIN's 29-group replicon training set.
Same data, same 5 folds, same kNN (k=5, cosine, distance-weighted).
Three rows: raw kNN (raw), validate_encoder_vs_knn.py encoder unchanged (britto), class-balanced encoder from run_encoder.py (gijs).
Each encoder gets the input its own pipeline uses."""
import sys, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

import validate_encoder_vs_knn as bx        # imported unchanged
import run_encoder as gd                    # class-balanced encoder

d = np.load(os.path.join(ROOT, "data", "inc_classifier.npz"), allow_pickle=True)
X_raw = d["X"].astype(np.float32)                                  # raw frequencies
X_l2 = X_raw / np.linalg.norm(X_raw, axis=1, keepdims=True)        # L2-normalised
yi = d["y"].astype(int)
names = d["group_names"].astype(str)
y = names[yi]
print(X_raw.shape, len(names), "groups", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
kw = dict(n_neighbors=5, metric="cosine", weights="distance")
P = {m: np.empty(len(y), dtype=object) for m in ["raw", "britto", "gijs"]}
folds = []

for f, (tr, te) in enumerate(cv.split(X_raw, y), 1):
    P["raw"][te] = KNeighborsClassifier(**kw).fit(X_raw[tr], y[tr]).predict(X_raw[te])

    mb = bx.train_encoder(X_raw[tr], yi[tr])                       # his defaults
    Ztr, Zte = bx.embed(mb, X_raw[tr]), bx.embed(mb, X_raw[te])
    P["britto"][te] = KNeighborsClassifier(**kw).fit(Ztr, y[tr]).predict(Zte)

    mg = gd.train(X_l2[tr], yi[tr], len(names), gd.CFGS["default"])
    Ztr, Zte = gd.embed(mg, X_l2[tr]), gd.embed(mg, X_l2[te])
    P["gijs"][te] = KNeighborsClassifier(**kw).fit(Ztr, y[tr]).predict(Zte)

    row = {m: f1_score(y[te], P[m][te].astype(str), average="macro") for m in P}
    folds.append(row)
    print(f"fold {f}: " + "  ".join(f"{m} {v:.3f}" for m, v in row.items()), flush=True)

out = {}
print(f"\n{'':8s} {'acc':>6s} {'macroF1':>8s} {'wF1':>6s} {'balacc':>7s}")
for m, p in P.items():
    p = p.astype(str)
    out[m] = dict(acc=accuracy_score(y, p), macroF1=f1_score(y, p, average="macro"),
                  weightedF1=f1_score(y, p, average="weighted"),
                  balacc=balanced_accuracy_score(y, p))
    o = out[m]
    print(f"{m:8s} {o['acc']:6.3f} {o['macroF1']:8.3f} {o['weightedF1']:6.3f} {o['balacc']:7.3f}")

wins = sum(r["gijs"] > r["britto"] for r in folds)
print(f"\nGijs beats Britto on macro F1 in {wins}/5 folds")
out["per_fold_macroF1"] = folds
json.dump(out, open(os.path.join(HERE, "results_head2head.json"), "w"), indent=1)
print("wrote results_head2head.json")
