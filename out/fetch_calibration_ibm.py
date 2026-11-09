"""fetch_calibration_ibm.py — calibration data at execution time for Table "tab:calib" =================================================================================== Run in the `qhermes` env (qiskit-ibm-runtime, account already saved):"""
import sys
import statistics as st
from qiskit_ibm_runtime import QiskitRuntimeService

N2Q = {"ibm_fez": 254, "ibm_marrakesh": 251, "ibm_kingston": 252}

def used_qubits(job):
    try:
        circ = job.inputs["pubs"][0][0]
        lay = circ.layout
        if lay is not None:
            return sorted(lay.final_index_layout())
        return sorted({circ.find_bit(q).index for inst in circ.data for q in inst.qubits})
    except Exception:
        return None

def main(job_ids):
    service = QiskitRuntimeService()
    for jid in job_ids:
        job = service.job(jid)
        name = job.backend().name
        props = job.properties()
        qs = used_qubits(job) or list(range(len(props.qubits)))
        t1 = [props.t1(q) * 1e6 for q in qs]
        t2 = [props.t2(q) * 1e6 for q in qs]
        ro = [props.readout_error(q) for q in qs]
        e2 = []
        for g in props.gates:
            if len(g.qubits) == 2 and set(g.qubits) <= set(qs):
                for p in g.parameters:
                    if p.name == "gate_error" and p.value < 1:
                        e2.append(p.value)
        m2 = st.median(e2) if e2 else float("nan")
        fid = (1 - m2) ** N2Q.get(name, 0)
        print(f"{name}  job {jid}  date {props.last_update_date}  qubits {qs}")
        print(f"  {name} & {m2:.2e} & {st.median(ro):.2e} & {st.median(t1):.0f} & "
              f"{st.median(t2):.0f} & {fid:.2f} & ... \\\\")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
