from configparser import MAX_INTERPOLATION_DEPTH
from curses import KEY_SOPTIONS
from re import L
import numpy as np
import opt_einsum as oe
from opt_einsum.helpers import compute_size_by_dict
import cotengra as ctg
from tn_qsim.mps import MPS
from tn_qsim.mpo import MPO
from tn_qsim.peps import PEPS
from tn_qsim.peps3D import PEPS3D
from scipy.stats import unitary_group, ortho_group
import time
import functools
import sys
import math
import os

args = sys.argv
settings = ["width", "height", "depth", "ctype", "mode", "cseed", "aseed", "minimize", "pseed", "nworkers", "gtype", "max_repeat", "calc_amp"]
ns  = locals()
print("settings")
for i, arg in enumerate(args):
    if i == 0:
        continue
    ns[settings[i-1]] = args[i]
    print(f"{settings[i-1]}: {args[i]}")

############ arg settings ############
width = int(width) # circuit width
height = int(height) # circuit height
depth = int(depth) # circuit total depth
ctype = ctype # circuit type, xyt only
mode = mode # Tensorcore mode, FP16TCEC, TF32TCEC, FP16TC, TF32TC, CUBLAS, CUBLAS64F
cseed = int(cseed) # seed for circuit
aseed = int(aseed) # seed for amplitude
minimize = minimize # target for hyperoptimizer, "gputime", "flops"
pseed = int(pseed) # seed for path
nworkers = int(nworkers) # number or parallelization
gtype = gtype # gpu_table type, just for print and contraction path
max_repeat = int(max_repeat) # config for cotengra
calc_amp = True if calc_amp == "True" else False # whether to calc amp

############ tensorcore mode settings ############

"""if mode == "CUBLAS":
    os.environ["LD_PRELOAD"] = ""
else:
    os.environ["LD_PRELOAD"] = "/data/cuMpSGEMM/hijack/lib/libcumpsgemm.so"""

if mode == "CUBLAS64F":
    os.environ["CUMPSGEMM_COMPUTE_MODE"] = "CUBLAS"
else:
    os.environ["CUMPSGEMM_COMPUTE_MODE"] = mode
    

############ path settings ############
#max_repeat = 512 # config for cotengra
max_time = 3600 # config for cotengra
target_size = 27 # config for cotengra
type = "optuna" # config for cotengra optuna, skopt, nevergrad

############ folder settings ############

data_folder = "data20220918_1"

qnum = width * height

start_time = time.time()
dir = "GRCS/inst/rectangular/cz_v2/"
file = dir + f"{width}x{height}/inst_{width}x{height}_{depth}_{cseed}.txt"

ctg_dir = f"ctg_TC_RCS_path_cache/20220918_1/targetsize{target_size}/{type}/minimize{minimize}/gtype_{gtype}/seed{pseed}"
os.makedirs(dir, exist_ok=True)


algorithm = ctg.ReusableHyperOptimizer(
    methods="kahypar",
    minimize=minimize,
    max_repeats=max_repeat,
    max_time=max_time,
    parallel=nworkers,
    slicing_reconf_opts={"target_size": 2**target_size},
    optlib=f"{type}",
    progbar=True,
    directory=ctg_dir,
)

#############

precision = "complex128" if mode == "CUBLAS64F" else "complex64"
np.random.seed(aseed)

gate_list = []
with open(file) as f:
    for idx, line in enumerate(f):
        if idx == 0:
            continue
        gate_list.append(line.split())

zero = np.array([1, 0])
one = np.array([0, 1])

tensors = [zero.reshape(2,1,1,1,1) for i in range(width * height)]
# use 3D contraction to exact amp calc
peps3D = PEPS3D(tensors, height, width)

x_1_2 = np.array([[1, -1j],[-1j, 1]]) / np.sqrt(2)
y_1_2 = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])

def from_gate2_to_tensor(gate2_tensor, bdim):
    U, s, Vh = np.linalg.svd(gate2_tensor.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4), full_matrices=False)
    U = U[:, :bdim]
    s = s[:bdim]
    Vh = Vh[:bdim, :]
    U = np.dot(U, np.diag(s)).reshape(2,2,1,bdim)
    Vh = Vh.T.reshape(2,2,bdim,1)
    return U, Vh

gate_idx = 0
uidx = 0

for idx, gate in enumerate(gate_list):
    # print("gate:", gate)
    gate_round = int(gate[0])

    # 2 qubit gate
    if gate[1] == "cz":
        apply_tensor = None
        apply_U = None
        apply_Vh = None
        bond_dim = None
        if ctype == "xyt":
            apply_tensor = CZ
            bond_dim = 2
        apply_U, apply_Vh = from_gate2_to_tensor(apply_tensor, bond_dim)
        mpo_gate = MPO([apply_U, apply_Vh])
        peps3D.apply_MPO([int(gate[2]), int(gate[3])], mpo_gate)
            
    else:
        # single qubit gate
        apply_tensor = None
        if gate[1] == "h":
            apply_tensor = H
        elif ctype == "xyt":
            if gate[1] == "t":
                apply_tensor = T
            elif gate[1] == "x_1_2":
                apply_tensor = x_1_2
            elif gate[1] == "y_1_2":
                apply_tensor = y_1_2
        mpo_gate = MPO([apply_tensor.reshape(2,2,1,1)])
        peps3D.apply_MPO([int(gate[2])], mpo_gate)

bitstring = "".join(np.random.choice(["0","1"]) for _ in range(qnum))
print(f"random bitstring : {bitstring}")

# calc exact amplitude
amp_tensors = [one if bitstring[q] == "1" else zero for q in range(qnum)]
tn, tree = peps3D.find_amplitude_tree_by_quimb(amp_tensors, algorithm=algorithm, visualize=True)
print(f"after simplification |V|: {tn.num_tensors}, |E|: {tn.num_indices}")
print(f"tree gpitime: {tree.total_gputime():,}, tree cost: {tree.total_flops():,}, sp_cost: {tree.max_size():,}")
print(f"slice: {tree.sliced_inds}".encode("utf-8").strip())

if calc_amp:
    start1 = time.time()
    amplitude = peps3D.amplitude_by_quimb(amp_tensors, tn=tn, tree=tree, gpu=True, backend="cupy", precision=precision)
    print(f"true prob: {np.real_if_close(amplitude.conj() * amplitude)}")
    end1 = time.time()
    print(f"elapsed time for contracting: {end1 - start1}")

# settings = ["width", "height", "depth", "ctype", "mode", "cseed", "aseed", "minimize", "pseed", "nworkers", "gtype", "max_repeat", "calc_amp"]
#data_folder = f"{data_folder}/width{width}/height{height}/depth{depth}/ctype{ctype}/cseed{cseed}/minimize{minimize}/pseed{pseed}/gtype{gtype}"
#os.makedirs(data_folder, exist_ok=True)
#np.save(f"{data_folder}/contraction_cost_{mode}_aseed{aseed}.npy", np.array(tree.total_flops()))
#np.save(f"{data_folder}/elapsed_time_{mode}_aseed{aseed}.npy", np.array(end1 - start1))
#np.save(f"{data_folder}/prob_{mode}_aseed{aseed}.npy", amplitude.conj() * amplitude)