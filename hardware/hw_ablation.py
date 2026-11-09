"""hw_ablation.py — controlled cross-vendor ablation for the QCDF pilot (reviewer point A2) ========================================================================================== Same SHALLOW circuit on IQM and IBM, back to back (same calibration window), with the calibration snapshot of each device saved next to the counts."""
import sys, json, time, datetime as dt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from dicke_scs import dicke_circuit

N, K, J = 6, 3, 0.3
v = np.array([0, .489, .310, 1, .484, .826]); c = np.array([.639, .308, 0, .878, .044, 1])
r = np.array([.005, .249, 0, 1, .308, .892]); u = np.array([.548, 1, .828, 0, .133, .043])
H = 1 * c + .5 * r + .3 * u - 2 * v
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
OPT = '011001'
ANGLES = {1: ([4.547208], [0.666897])}

ANGLES_P4 = None
SHOTS = 4000

def circuit(gammas, betas):
    qc = QuantumCircuit(N)
    qc.compose(dicke_circuit(N, K), range(N), inplace=True)
    for g, b in zip(gammas, betas):
        for i in range(N): qc.rz(-g * H[i], i)
        for (i, j) in EDGES: qc.rzz(2 * g * J, i, j)
        for (i, j) in EDGES: qc.append(XXPlusYYGate(2 * b), [i, j])
    qc.measure_all()
    return qc

def summarize(counts):
    tot = sum(counts.values())
    key = lambda s: s.replace(' ', '')[::-1]
    feas = sum(n for s, n in counts.items() if key(s).count('1') == K)
    opt = sum(n for s, n in counts.items() if key(s) == OPT)
    pu = 1 / (220 if WINDOW else 20)
    return dict(shots=tot, P_feas=feas / tot, P_opt=opt / tot, P_opt_ps=opt / max(feas, 1),
                A=opt / tot / pu, A_ps=opt / max(feas, 1) / pu)

WINDOW = None
def window_circuits():
    w = json.load(open('hw_window.json')); sd = np.std(w['E_feasible'])
    n = len(w['heff']); hq = np.array(w['heff']) / sd
    Jq = {(min(a, b), max(a, b)): v / sd for a, b, v in w['Jin']}
    global N, K, OPT, EDGES
    N, K, OPT, EDGES = n, w['k'], w['opt'], [tuple(e) for e in w['edges']]
    def circ(gs, bs):
        qc = QuantumCircuit(N); qc.compose(dicke_circuit(N, K), range(N), inplace=True)
        for g, b in zip(gs, bs):
            for i in range(N): qc.rz(-g * (hq[i] + sum(v for (a, c), v in Jq.items() if i in (a, c)) / 2), i)
            for (i, j), v in Jq.items():
                if abs(v) > 1e-9: qc.rzz(g * v / 2, i, j)
            for (i, j) in EDGES: qc.append(XXPlusYYGate(2 * b), [i, j])
        qc.measure_all(); return qc
    a1, a2 = w['angles_p1'], w['angles_p2']
    return {'win_p1': circ(a1[:1], a1[1:]), 'win_p2': circ(a2[:2], a2[2:])}

def circuits():
    if WINDOW: return window_circuits()
    out = {'p1': circuit(*ANGLES[1])}
    if ANGLES_P4 is not None: out['p4'] = circuit(*ANGLES_P4)
    return out

def ibm_snapshot(backend, qubits):
    snap = {}
    try:
        props = backend.properties()
        snap['last_update'] = str(props.last_update_date)
        snap['T1_us'] = {q: props.t1(q) * 1e6 for q in qubits}
        snap['T2_us'] = {q: props.t2(q) * 1e6 for q in qubits}
        snap['readout_error'] = {q: props.readout_error(q) for q in qubits}
        snap['2q_error'] = {str(g.qubits): p.value for g in props.gates if len(g.qubits) == 2 and set(g.qubits) <= set(qubits)
                            for p in g.parameters if p.name == 'gate_error'}
    except Exception as e:
        snap['error'] = repr(e)
    return snap

def run_ibm(name):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(); backend = svc.backend(name)
    res = dict(backend=name, utc=dt.datetime.utcnow().isoformat(), runs={})
    for lab, qc in circuits().items():
        t = transpile(qc, backend, optimization_level=3, seed_transpiler=1)
        qubits = sorted(t.layout.final_index_layout()) if t.layout else list(range(N))
        snap = ibm_snapshot(backend, qubits)
        job = SamplerV2(mode=backend).run([t], shots=SHOTS)
        counts = job.result()[0].data.meas.get_counts()
        res['runs'][lab] = dict(job_id=job.job_id(), depth=t.depth(), n2q=t.num_nonlocal_gates(), qubits=qubits,
                                calibration=snap, counts=counts, **summarize(counts))
        print(lab, name, job.job_id(), {k: v for k, v in res['runs'][lab].items() if k.startswith(('P_', 'A', 'depth', 'n2q'))})
    return res

def run_iqm(url):
    import os
    from iqm.qiskit_iqm import IQMProvider
    backend = IQMProvider(url, token=os.environ.get('IQM_TOKEN')).get_backend()
    res = dict(backend=url, utc=dt.datetime.utcnow().isoformat(), runs={})
    for lab, qc in circuits().items():
        t = transpile(qc, backend, optimization_level=3, seed_transpiler=1)
        job = backend.run(t, shots=SHOTS); result = job.result(); counts = result.get_counts()
        meta = {}
        try:
            cs = result._metadata.get('calibration_set_id') if hasattr(result, '_metadata') else None
            meta['calibration_set_id'] = str(cs or getattr(job, 'calibration_set_id', None))
            client = backend.client
            meta['quality_metrics'] = json.loads(json.dumps(client.get_quality_metric_set(cs).model_dump(), default=str))
        except Exception as e:
            meta['note'] = ('copy median CZ fidelity, readout fidelity, T1, T2 of the used qubits from IQM Resonance '
                            '(job page -> calibration set); automatic fetch failed: ' + repr(e))
        res['runs'][lab] = dict(job_id=str(job.job_id()), depth=t.depth(), n2q=t.num_nonlocal_gates(),
                                qubits=[t.find_bit(q).index for q in t.qubits], calibration=meta, counts=counts, **summarize(counts))
        print(lab, url, job.job_id(), {k: v for k, v in res['runs'][lab].items() if k.startswith(('P_', 'A', 'depth', 'n2q'))})
    return res

def dry():
    from qiskit.quantum_info import Statevector
    for lab, qc in circuits().items():
        qq = qc.remove_final_measurements(inplace=False)
        pr = Statevector(qq).probabilities_dict()
        counts = {s: int(round(p * SHOTS)) for s, p in pr.items()}
        t = transpile(qc, basis_gates=['cz', 'rz', 'sx', 'x'], coupling_map=[[i, i + 1] for i in range(N - 1)] + [[i + 1, i] for i in range(N - 1)],
                      optimization_level=3, seed_transpiler=1)
        print(lab, 'noiseless', summarize(counts), 'linear-map depth', t.depth(), '2Q', t.num_nonlocal_gates())

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dry'
    WINDOW = 'window' in sys.argv
    if mode == 'dry': dry(); sys.exit()
    res = run_ibm(sys.argv[2]) if mode == 'ibm' else run_iqm(sys.argv[2])
    fn = f"ablation_{res['backend'].split('/')[-1]}_{int(time.time())}.json"
    json.dump(res, open(fn, 'w'), indent=1, default=str); print('saved', fn)
