import os
import numpy as np
import gdown
from fenics import IntervalMesh

from dlroms.roms import snapshots
from dlroms.cores import CPU
import dlroms.fespaces as fe

# --- SETUP ---
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- NAVIER-STOKES ---
print("\n--- DOWNLOADING NAVIER-STOKES DATASET ---")
gdown.download(id = "1Sj7sMVgrOv5dM4wYbiGCyDktFpdk8VGR", output = f"{DATA_DIR}/nstokes_data.npz")
gdown.download(id = "1pkKBI4qGB2C6IGiXgAiY5CvZx0DEm88O", output = f"{DATA_DIR}/nstokes_mesh.xml")

# --- GAUSSIAN ---
print("\n--- GENERATING GAUSSIAN DATASET ---")

# Mesh
mesh = IntervalMesh(1000, 0, 1)
fe.savemesh(path=f"{DATA_DIR}/gaussian_mesh.xml", mesh=mesh)
Vh = fe.space(mesh, 'CG', 1)

# Data solver definition
def solver(seed):
    np.random.seed(seed)
    x = fe.dofs(Vh)[:, 0]

    mu = np.random.rand(4)

    u = mu[2]*np.exp(-100*(x-mu[0])**2) + mu[3]*np.exp(-100*(x-mu[1])**2)

    return mu, u

# Snapshots saving
snapshots(n=1000, sampler=solver, core=CPU, filename=f"{DATA_DIR}/gaussian_data.npz")

print(f"\n[SUCCESS] All datasets have been correctly saved to '{DATA_DIR}/'")
