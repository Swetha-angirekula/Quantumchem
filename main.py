from fastapi import FastAPI
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "LiH VQE API is running on Render!"}

@app.get("/vqe")
def run_vqe():
    # Constants
    ANGSTROM_TO_BOHR = 1.8897259886
    symbols = ["Li", "H"]
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.6 * ANGSTROM_TO_BOHR],
    ])

    # Molecule
    molecule = qchem.Molecule(symbols, coordinates, charge=0, mult=1, basis_name="sto-3g")
    active_electrons, active_orbitals = 4, 4
    hamiltonian, num_qubits = qchem.molecular_hamiltonian(
        molecule, active_electrons=active_electrons, active_orbitals=active_orbitals
    )
    hf_state = qchem.hf_state(active_electrons, num_qubits)
    singles, doubles = qchem.excitations(active_electrons, num_qubits)
    s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("default.qubit", wires=num_qubits)
    from pennylane.templates.subroutines import UCCSD

    @qml.qnode(dev)
    def circuit(params):
        UCCSD(params, wires=range(num_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
        return qml.expval(hamiltonian)

    # Optimization
    params = np.zeros(len(s_wires) + len(d_wires), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=0.1)
    energy_progress = []
    for n in range(50):  # fewer steps for speed
        params, energy = optimizer.step_and_cost(circuit, params)
        energy_progress.append(float(energy))

    final_energy = float(energy)

    # Exact diagonalization
    Hmat = qml.utils.sparse_hamiltonian(hamiltonian).toarray()
    eigvals, _ = np.linalg.eigh(Hmat)
    exact_energy = float(np.min(eigvals))

    return {
        "final_energy": final_energy,
        "exact_energy": exact_energy,
        "convergence": energy_progress
    }
