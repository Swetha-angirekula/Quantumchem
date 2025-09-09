import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Constants
ANGSTROM_TO_BOHR = 1.8897259886

# Define LiH molecule symbols and coordinates (converted to Bohr units)
symbols = ["Li", "H"]
coordinates = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.6 * ANGSTROM_TO_BOHR],  # Li-H bond length in Bohr (~1.6 Å)
])

# Create Molecule object with basis set
molecule = qchem.Molecule(
    symbols=symbols,
    coordinates=coordinates,
    charge=0,
    mult=1,
    basis_name="sto-3g"
)

# Active electrons and orbitals (reducing qubit count)
# LiH has 4 electrons total. We'll consider 2 active electrons in 2 orbitals (minimal active space)
active_electrons = 4
active_orbitals = 4

# Build Hamiltonian
hamiltonian, num_qubits = qchem.molecular_hamiltonian(
    molecule,
    active_electrons=active_electrons,
    active_orbitals=active_orbitals
)

print("Number of qubits:", num_qubits)
print("Number of Hamiltonian terms:", len(hamiltonian))

# Prepare Hartree-Fock state
hf_state = qchem.hf_state(active_electrons, num_qubits)

# Get single and double excitations
singles, doubles = qchem.excitations(active_electrons, num_qubits)

# Convert excitations to wire format
s_wires, d_wires = qchem.excitations_to_wires(singles, doubles)

# Define quantum device
dev = qml.device("default.qubit", wires=num_qubits)

from pennylane.templates import UCCSD

# Define VQE circuit
@qml.qnode(dev)
def circuit(params):
    UCCSD(params, wires=range(num_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
    return qml.expval(hamiltonian)

# Initialize parameters
optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
params = np.zeros(len(s_wires) + len(d_wires), requires_grad=True)

# Variables to track energy convergence
previous_params = None
energy_progress = []

def store_intermediate_result(params, energy):
    global previous_params
    if previous_params is None or not np.allclose(previous_params, params):
        energy_progress.append(energy)
        previous_params = params.copy()
        print(f"Tracked Energy: {energy:.6f}")

# Optimization loop with tracking
max_iters = 150
for n in range(max_iters):
    params, energy = optimizer.step_and_cost(circuit, params)
    store_intermediate_result(params, energy)
    if n % 10 == 0:
        print(f"Step {n}, Params norm: {np.linalg.norm(params):.6f}")
        print(f"Step {n}: Energy = {energy:.6f} Hartree")

print("Final VQE energy (Hartree):", energy)

# Exact diagonalization for comparison
Hmat = qml.matrix(hamiltonian)
eigvals, _ = np.linalg.eigh(Hmat)
print("Exact ground state energy (Hartree):", np.min(eigvals))

# Plot energy convergence
plt.figure(figsize=(8, 5))
plt.plot(range(len(energy_progress)), energy_progress, marker='o')
plt.xlabel("Optimizer Call Count (Unique Parameter Updates)")
plt.ylabel("Energy (Hartree)")
plt.title("VQE Energy Convergence for LiH")
plt.grid(True)
plt.show()
