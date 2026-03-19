"""
QSVR - Quantum Support Vector Regressor
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
C_REG = 1.0
EPSILON = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def qsvr_predict(K, y, C=C_REG, eps=EPSILON, max_iter=500, tol=1e-6):
    n = len(y)
    alpha = np.zeros(n)
    alpha_star = np.zeros(n)

    for iteration in range(max_iter):
        alpha_old = alpha.copy()
        alpha_star_old = alpha_star.copy()

        for i in range(n):
            f_i = float(K[i] @ (alpha - alpha_star))
            # alpha update
            delta_a = (y[i] - f_i - eps)
            alpha[i] = np.clip(alpha[i] + 0.01 * delta_a, 0, C)
            # alpha_star update
            delta_as = (f_i - y[i] - eps)
            alpha_star[i] = np.clip(alpha_star[i] + 0.01 * delta_as, 0, C)

        if (np.max(np.abs(alpha - alpha_old)) < tol and
                np.max(np.abs(alpha_star - alpha_star_old)) < tol):
            break

    w = alpha - alpha_star
    pred = K @ w

    return pred


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QSVR po pozicijama (C={C_REG}, eps={EPSILON}) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = qsvr_predict(K, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QSVR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QSVR po pozicijama (C=1.0, eps=0.01) ---
  Poz 1 [1-33]: 1:0.146 | 2:0.127 | 3:0.111
  Poz 2 [2-34]: 8:0.088 | 5:0.077 | 9:0.076
  Poz 3 [3-35]: 13:0.075 | 12:0.073 | 14:0.071
  Poz 4 [4-36]: 23:0.074 | 21:0.073 | 18:0.072
  Poz 5 [5-37]: 29:0.075 | 26:0.073 | 27:0.072
  Poz 6 [6-38]: 33:0.096 | 32:0.093 | 35:0.091
  Poz 7 [7-39]: 7:0.186 | 38:0.153 | 37:0.130

==================================================
Predikcija (QSVR, deterministicki, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================
"""



"""
QSVR - Quantum Support Vector Regressor

Isti kvantni kernel kao QKR (ZZFeatureMap, fidelity, 5 qubita)
Razlika: umesto ridge regresije, koristi SVR algoritam sa epsilon-insensitive loss
Dual koordinatni descent za optimizaciju alpha koeficijenata (C=1.0, eps=0.01)
Egzaktno, deterministicki, bez shots-a
Brz kao QKR - kernel se racuna jednom, SVR konvergira brzo na 32 tacke
"""
