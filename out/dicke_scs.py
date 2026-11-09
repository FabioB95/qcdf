"""dicke_scs.py — Low-depth deterministic Dicke-state preparation ================================================================ Baertschi & Eidenbenz (2019), "Deterministic preparation of Dicke states", split-and-cyclic-shift (SCS) construction: O(n k) CNOTs, depth O(n), no ancillas. Drop-in replacement for qc.initialize(dicke_vec(N,K), ...) / StatePreparation, which Qiskit decomposes into a generic (much deeper) isometry."""
import numpy as np
from math import comb
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

def _cry(qc, theta, c, t):
    qc.ry(theta / 2, t); qc.cx(c, t); qc.ry(-theta / 2, t); qc.cx(c, t)

def _ccry(qc, theta, c1, c2, t):
    a = theta / 4
    qc.ry(a, t); qc.cx(c2, t); qc.ry(-a, t); qc.cx(c1, t)
    qc.ry(a, t); qc.cx(c2, t); qc.ry(-a, t); qc.cx(c1, t)

def _scs(qc, qubits, n, k):
    q = lambda m: qubits[m - 1]

    qc.cx(q(n - 1), q(n))
    _cry(qc, 2 * np.arccos(np.sqrt(1 / n)), q(n), q(n - 1))
    qc.cx(q(n - 1), q(n))

    for l in range(2, k + 1):
        qc.cx(q(n - l), q(n))
        _ccry(qc, 2 * np.arccos(np.sqrt(l / n)), q(n), q(n - l + 1), q(n - l))
        qc.cx(q(n - l), q(n))

def dicke_circuit(n, k):
    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    qubits = list(range(n))
    for m in range(n - k, n):
        qc.x(m)
    for l in range(n, k, -1):
        _scs(qc, qubits[:l], l, k)
    for l in range(k, 1, -1):
        _scs(qc, qubits[:l], l, l - 1)
    return qc

def dicke_vec(n, k):
    v = np.zeros(2 ** n)
    for x in range(2 ** n):
        if bin(x).count("1") == k:
            v[x] = 1
    return v / np.linalg.norm(v)

if __name__ == "__main__":
    for (n, k) in [(4, 2), (6, 3), (6, 2), (8, 4), (12, 6)]:
        sv = Statevector(dicke_circuit(n, k)).data
        fid = abs(np.vdot(dicke_vec(n, k), sv)) ** 2
        print(f"n={n:2d} k={k}  fidelity with |D_k^n> = {fid:.12f}")
        assert fid > 1 - 1e-9

    basis = ["cz", "rz", "sx", "x", "r"]
    line6 = [[i, i + 1] for i in range(5)] + [[i + 1, i] for i in range(5)]
    print("\nN=6, k=3 Dicke preparation only:")
    for name, qc in [("SCS (this file)", dicke_circuit(6, 3)),
                     ("initialize()", None)]:
        if qc is None:
            qc = QuantumCircuit(6); qc.initialize(dicke_vec(6, 3), range(6))
        for cmap_name, cmap in [("all-to-all", None), ("line", line6)]:
            t = transpile(qc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cmap,
                          optimization_level=3, seed_transpiler=1)
            print(f"  {name:16s} {cmap_name:10s} 2Q={t.count_ops().get('cz', 0):4d}  depth={t.depth():4d}")
