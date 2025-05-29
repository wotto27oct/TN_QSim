import numpy as np
from tn_qsim.mps import MPS
from tn_qsim.mpo import MPO
from tn_qsim.utils import *
from utils_repetition_code import *
import copy
import sys
import os
import scipy.stats as stats
import time
import tqdm

from pymatching import Matching
from get_kraus import *

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(linewidth=500)

###### config ######
args = sys.argv
settings = ["ncolumn", "nround", "theta", "gamma", "tau", "temp", "spread", "rtype", "init_state", "seed", "nshot", "threshold_err", "save_folder"]
ns  = locals()
print("settings")
for i, arg in enumerate(args):
    if i == 0:
        continue
    ns[settings[i-1]] = args[i]
    print(f"{settings[i-1]}: {args[i]}")

########################
ncolumn = int(ncolumn) # number of columns, must be odd
nround = int(nround)
theta = float(theta) # rotation noise strength, [0, np.pi]
gamma = float(gamma) # coupling strength to bath, typically 0.1 [MHz]
tau = float(tau) # the time of the thermalization process, typically 1.0 [us]
temp = float(temp) # temperature. typically 10[mK] ~ 100[mK]
spread = float(spread) # leakage spread
rtype = rtype # removal type, "noreset", "mlr", "dqlr"
init_state = int(init_state) # 0 or 1
seed = int(seed)
nshot = int(nshot)
threshold_err = int(threshold_err)
save_folder = save_folder
########################

stype = "mps"

if save_folder != "None" and save_folder != "False":
    if os.path.exists(f"{save_folder}/data/ncolumn{ncolumn}/nround{nround}/seed{seed}/theta{theta}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_classical_logical_error_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy"):
        print("result exists.")
        exit()

is_print = False
is_print_large = False
print_shot = 100

np.random.seed(seed)

qnum = 3*(2*ncolumn-1) + 2

pdim = 3 # physical dimension, >=3

basis = []
for i in range(pdim):
    bs_arr = np.zeros(pdim).astype(dtype=np.complex128)
    bs_arr[i] = 1.0
    basis.append(bs_arr)

Proj = [oe.contract("i,j->ij",basis[i],basis[i].conj()).reshape(pdim,pdim,1,1) for i in range(pdim)]
ResetProj = [oe.contract("i,j->ij",basis[0],basis[i].conj()).reshape(pdim,pdim,1,1) for i in range(pdim)]
NoResetProj = ResetProj[:2] + Proj[2:]

# thermal excitation
kraus_list = krauslist_by_gamma_tau_temp(gamma, tau, temp)

# leakage spread between 02<->22, 12<->22, 20<->22, 21<->22
R_02_22 = np.eye(9)
R_02_22[2,2] = R_02_22[8,8] = np.cos(spread * np.pi / 2.0)
R_02_22[2,8] = -np.sin(spread * np.pi / 2.0)
R_02_22[8,2] = np.sin(spread * np.pi / 2.0)

R_12_22 = np.eye(9)
R_12_22[5,5] = R_12_22[8,8] = np.cos(spread * np.pi / 2.0)
R_12_22[5,8] = -np.sin(spread * np.pi / 2.0)
R_12_22[8,5] = np.sin(spread * np.pi / 2.0)

R_20_22 = np.eye(9)
R_20_22[6,6] = R_20_22[8,8] = np.cos(spread * np.pi / 2.0)
R_20_22[6,8] = -np.sin(spread * np.pi / 2.0)
R_20_22[8,6] = np.sin(spread * np.pi / 2.0)

R_21_22 = np.eye(9)
R_21_22[7,7] = R_21_22[8,8] = np.cos(spread * np.pi / 2.0)
R_21_22[7,8] = -np.sin(spread * np.pi / 2.0)
R_21_22[8,7] = np.sin(spread * np.pi / 2.0)

R_spread = np.dot(np.dot(R_21_22, R_20_22), np.dot(R_12_22, R_02_22))
#print(R_spread)
R_spread_mpo = return_mpo_from_2qubit_op(R_spread, dim1=pdim, dim2=pdim)

# prepare long-range 
cross_tensor = oe.contract("ab,cd->abcd", np.eye(3), np.eye(R_spread_mpo.tensors[0].shape[3])).astype(np.complex128)
R_spread_mpo2 = R_spread_mpo
R_spread_mpo3 = MPO([R_spread_mpo2.tensors[0],cross_tensor,R_spread_mpo2.tensors[1]])
R_spread_mpo4 = MPO([R_spread_mpo2.tensors[0],cross_tensor,cross_tensor,R_spread_mpo2.tensors[1]])
R_spread_mpo5 = MPO([R_spread_mpo2.tensors[0],cross_tensor,cross_tensor,cross_tensor,R_spread_mpo2.tensors[1]])
R_spread_mpo6 = MPO([R_spread_mpo2.tensors[0],cross_tensor,cross_tensor,cross_tensor,cross_tensor,R_spread_mpo2.tensors[1]])
R_spread_mpo_list = [None, None, R_spread_mpo2, R_spread_mpo3, R_spread_mpo4, R_spread_mpo5, R_spread_mpo6]

# leakage removal
LeakageISWAP = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1j,0,0],[0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,1j,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]).astype(np.complex128)
LeakageISWAP_mpo = return_mpo_from_2qubit_op(LeakageISWAP, dim1=pdim, dim2=pdim)

H = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0], [1/np.sqrt(2), -1/np.sqrt(2), 0], [0, 0, 1.0]])
mpoX = MPO([X.reshape(3,3,1,1)])
mpoZ = MPO([Z.reshape(3,3,1,1)])
mpoH = MPO([H.reshape(3,3,1,1)])

cz_phi = 0.5
CZ = np.diag(np.array([1,1,1,1,-1,np.exp(1j*cz_phi*np.pi),1,np.exp(1j*cz_phi*np.pi),1]))
#CZ = np.diag(np.array([1,1,1,1,-1,1,1,1,1]))
CZ_mpo = return_mpo_from_2qubit_op(CZ, dim1=pdim, dim2=pdim)

# prepare long-range CZ
cross_tensor = oe.contract("ab,cd->abcd", np.eye(3), np.eye(3)).astype(np.complex128)
CZ_mpo2 = CZ_mpo
CZ_mpo3 = MPO([CZ_mpo2.tensors[0],cross_tensor,CZ_mpo2.tensors[1]])
CZ_mpo4 = MPO([CZ_mpo2.tensors[0],cross_tensor,cross_tensor,CZ_mpo2.tensors[1]])
CZ_mpo5 = MPO([CZ_mpo2.tensors[0],cross_tensor,cross_tensor,cross_tensor,CZ_mpo2.tensors[1]])
CZ_mpo6 = MPO([CZ_mpo2.tensors[0],cross_tensor,cross_tensor,cross_tensor,cross_tensor,CZ_mpo2.tensors[1]])
CZ_mpo_list = [None, None, CZ_mpo2, CZ_mpo3, CZ_mpo4, CZ_mpo5, CZ_mpo6]


def initialize_Urot():
    # fix rotation
    phi = np.ones(qnum) * np.pi / 2
    lam = np.random.uniform(low=0.0, high=2*np.pi, size=qnum)
    varphi = np.random.uniform(low=0.0, high=2*np.pi, size=qnum)
    Urot = []
    for i in range(qnum):
        Urot.append(Rot(theta*np.pi, phi[i], lam[i], varphi[i]).reshape(pdim,pdim,1,1))
    return Urot

def apply_H_to_zcheck(mps):
    mps.apply_MPO([0], MPO(mpoH.tensors))
    for pos in range(5, qnum, 6):
        mps.apply_MPO([pos], MPO(mpoH.tensors))
    mps.apply_MPO([qnum-1], MPO(mpoH.tensors))

def apply_H_to_xcheck(mps):
    for pos in range(4, qnum-3, 6):
        mps.apply_MPO([pos], MPO(mpoH.tensors))
        mps.apply_MPO([pos+2], MPO(mpoH.tensors))

def apply_Zlogical(mps):
    for pos in [1, 2, 3]:
        mps.apply_MPO([pos], MPO(mpoX.tensors))

def apply_Xlogical(mps):
    for pos in range(3, 3+6*(ncolumn-1)+1, 6):
        mps.apply_MPO([pos], MPO(mpoZ.tensors))

def prepare_right_tensors(mps):
    right_trace_tensors = [np.array([[1]])] # Warn! ith tensor: right tensor from qnum-1-i
    for pos in range(qnum-1, -1, -1):
        if pos == qnum-1:
            # suppose right bond dim = 1
            right_tensor = oe.contract("abc,aBC->bBcC",mps.tensors[pos],mps.tensors[pos].conj())
            left_bdim = right_tensor.shape[0]
            right_trace_tensors.append(right_tensor.reshape(left_bdim, left_bdim))
        else:
            right_trace_tensors.append(oe.contract("abc,aBC,cC->bB",mps.tensors[pos],mps.tensors[pos].conj(),right_trace_tensors[-1]))
    return right_trace_tensors

def apply_kraus(mps, kraus_list):
    # apply amplitude damping channel error for each data and ancilla
    right_trace_tensors = prepare_right_tensors(mps)  # Warn! ith tensor: right tensor from qnum-1-i
    left_tensor = np.array([[1]])
    for pos in range(0, qnum):
        right_tensor = right_trace_tensors[qnum-1-pos]
        pr_kraus_list = []
        left_tensor_kraus_list = []
        for K in kraus_list:
            left_tensor_kraus = oe.contract("bB,abc,ABC,da,dA->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj(),K,K.conj())
            pr = oe.contract("cC,cC",left_tensor_kraus,right_tensor)
            left_tensor_kraus_list.append(left_tensor_kraus)
            pr_kraus_list.append(pr)
        pr_kraus_list = np.real_if_close(pr_kraus_list)
        np.putmask(pr_kraus_list, np.abs(pr_kraus_list) < 1e-8, 0)
        # assert np.allclose(np.sum(pr_kraus_list), 1.0)
        pr_kraus_list /= np.sum(pr_kraus_list)

        chosen_kraus_index = np.random.choice([i for i in range(len(kraus_list))], p=pr_kraus_list)
        mps.apply_MPO([pos], MPO([kraus_list[chosen_kraus_index].reshape(3,3,1,1)]), is_normalize=False)
        mps.tensors[pos] /= np.sqrt(pr_kraus_list[chosen_kraus_index])
        left_tensor = left_tensor_kraus_list[chosen_kraus_index] / pr_kraus_list[chosen_kraus_index]

def apply_cz_and_Urot(mps, Urot, sup):
    fidelity = mps.apply_MPO(sup, MPO(CZ_mpo_list[len(sup)].tensors), is_truncate=True)
    for q in sup:
        mps.apply_MPO([q], MPO([Urot[q]]))
    return fidelity

def apply_leakage_spread(mps, sup):
    return mps.apply_MPO(sup, MPO(R_spread_mpo_list[len(sup)].tensors), is_truncate=True)

def return_support(order):
    # xcheck, zcheck, xcheck, xcheck, zcheck, xcheck
    # edge_sup: for leftmost and rightmost z-stabilizer
    # pos: 1, 13, 25,... (the index of top-left qubit for each square)
    # support: the support of stabilizer measurement, in order ancilla appears on MPS
    if order == 0:
        # top-right
        edge_sup = [1, 0]
        support = lambda pos: [[pos+3, pos+4, pos+5, pos+6, pos+7, pos+8], [pos+7, pos+6, pos+5, pos+4], [pos+5, pos+6], [pos+9, pos+10, pos+11, pos+12, pos+13], [pos+12, pos+11, pos+10], None]
    elif order == 1:
        # down-right
        edge_sup = [2, 1, 0]
        support = lambda pos: [None, [pos+8, pos+7, pos+6, pos+5, pos+4], [pos+5, pos+6, pos+7], [pos+9, pos+10, pos+11, pos+12, pos+13, pos+14], [pos+13, pos+12, pos+11, pos+10], [pos+11, pos+12]]
    elif order == 2:
        # top-left
        edge_sup = [qnum-3, qnum-2, qnum-1]
        support = lambda pos: [[pos+3, pos+2], [pos+1, pos+2, pos+3, pos+4], [pos+5, pos+4, pos+3, pos+2, pos+1, pos], [pos+9, pos+8, pos+7], [pos+6, pos+7, pos+8, pos+9, pos+10], None]
    elif order == 3:
        # down-left
        edge_sup = [qnum-2, qnum-1]
        support = lambda pos: [None, [pos+2, pos+3, pos+4], [pos+5, pos+4, pos+3, pos+2, pos+1], [pos+9, pos+8], [pos+7, pos+8, pos+9, pos+10], [pos+11, pos+10, pos+9, pos+8, pos+7, pos+6]]
    return edge_sup, support

def apply_syndrome_operator(mps, Urot):
    fidelity = 1.0
    apply_H_to_xcheck(mps)
    apply_H_to_zcheck(mps)
    for order in range(4):
        edge_sup, support = return_support(order)
        # apply z-stab meas in leftmost and rightmost
        fidelity *= apply_cz_and_Urot(mps, Urot, edge_sup)
        if np.linalg.norm(spread) > 0.001:
            fidelity *= apply_leakage_spread(mps, edge_sup)
        for pos in range(1, qnum-4, 12):
            support_list = support(pos)
            for idx, sup in enumerate(support_list):
                if sup is None:
                    continue
                if idx in [0, 2, 3, 5]:
                    # apply H to data qubit in x-check   
                    mps.apply_MPO([sup[-1]], MPO(mpoH.tensors))
                fidelity *= apply_cz_and_Urot(mps, Urot, sup)
                if np.linalg.norm(spread) > 0.001:
                    fidelity *= apply_leakage_spread(mps, sup)
                if idx in [0, 2, 3, 5]:
                    # apply H to data qubit in x-check   
                    mps.apply_MPO([sup[-1]], MPO(mpoH.tensors))
                fidelity *= mps.canonicalization(threshold=1-0.1**threshold_err)
    apply_H_to_xcheck(mps)
    apply_H_to_zcheck(mps)
    return fidelity

def measure_qubit(mps, pos, left_tensor, right_tensor, is_postselect=False, stab_type="None", is_apply_mpo=True):
    pr_list = []
    left_tensor_list = []
    for i in range(pdim):
        left_tensor_list.append(oe.contract("bB,abc,ABC,aA->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj(),Proj[i].reshape(3,3)))
        pr_list.append(oe.contract("cC,cC",left_tensor_list[-1],right_tensor))
    pr_list = np.real_if_close(pr_list)
    np.putmask(pr_list, np.abs(pr_list) < 1e-8, 0)
    pr_list /= np.sum(pr_list)

    if is_print:
        print(f"pos {pos} {stab_type} syndrome pr0={pr_list[0]}, pr1={pr_list[1]} pr2={pr_list[2]}")

    if is_apply_mpo:
        if not is_postselect:
            prob_random_value = np.random.uniform()
        else:
            prob_random_value = 0.0
        for i in range(pdim):
            if np.sum(pr_list[:i+1]) > prob_random_value:
                if is_print:
                    print(f"{i} selected")
                if rtype == "noreset":
                    mps.apply_MPO([pos], MPO([NoResetProj[i]]), is_normalize=False)
                else:
                    mps.apply_MPO([pos], MPO([ResetProj[i]]), is_normalize=False)
                mps.tensors[pos] /= np.sqrt(pr_list[i])
                left_tensor = left_tensor_list[i] / pr_list[i]
                if i < 2:
                    syndrome = i
                else:
                    syndrome = 0 if np.random.uniform() < 0.5 else 1
                break
    else:
        # just calculate trace
        left_tensor = oe.contract("bB,abc,aBC->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj())
        syndrome = None
    return left_tensor, pr_list, syndrome

def measure_ancilla(mps, is_postselect=False):
    syndrome_round = []
    prob_list_round = []

    right_trace_tensors = prepare_right_tensors(mps) # # Warn! ith tensor: right tensor from qnum-1-i
    left_tensor = np.array([[1]])
    for pos in range(qnum):
        if pos != 0 and pos != qnum-1 and ((pos - 1) // 3) % 2 == 0:
            # data qubit
            left_tensor = oe.contract("bB,abc,aBC->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj())
        else:
            # ancilla qubit
            if (pos - 1) % 3 == 1 or pos == 0 or pos == qnum-1:
                stab_type = "xstab"
            else:
                stab_type = "zstab"
            right_tensor = right_trace_tensors[qnum-1-(pos)]
            left_tensor, pr_list, syndrome = measure_qubit(mps, pos, left_tensor, right_tensor, is_postselect, stab_type)
            prob_list_round.append(pr_list)
            syndrome_round.append(syndrome)
            
    return syndrome_round, prob_list_round

def measure_data(mps, is_postselect=False):
    syndrome_round = []
    prob_list_round = []

    right_trace_tensors = prepare_right_tensors(mps) # # Warn! ith tensor: right tensor from qnum-1-i
    left_tensor = np.array([[1]])
    for pos in range(qnum):
        if pos == 0 or pos == qnum-1 or ((pos - 1) // 3) % 2 == 1:
            # ancilla qubit
            left_tensor = oe.contract("bB,abc,aBC->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj())
        else:
            # data qubit
            right_tensor = right_trace_tensors[qnum-1-(pos)]
            left_tensor, pr_list, syndrome = measure_qubit(mps, pos, left_tensor, right_tensor, is_postselect, stab_type="final-data")
            prob_list_round.append(pr_list)
            syndrome_round.append(syndrome)
            
    return syndrome_round, prob_list_round

def execute():
    # fix rotation
    phi = np.ones(qnum) * np.pi / 2
    lam = np.random.uniform(low=0.0, high=2*np.pi, size=qnum)
    varphi = np.random.uniform(low=0.0, high=2*np.pi, size=qnum)
    Urot = []
    for i in range(qnum):
        Urot.append(Rot(theta, phi[i], lam[i], varphi[i]).reshape(3,3,1,1))

    # prepare MPS
    mps = MPS([zero.reshape(3,1,1) for _ in range(qnum)], threshold_err=0.1**threshold_err)
    mps.canonicalization()
    fidelity = 1.0

    # apply H to all data qubit to encode |+> state
    for pos in range(qnum):
        if pos == 0 or pos == qnum-1 or ((pos - 1) // 3) % 2 == 1:
            # ancilla qubit
            continue
        else:
            mps.apply_MPO([pos], MPO(mpoH.tensors))

    # encode logical qubit
    apply_kraus(mps, kraus_list=kraus_list)
    fidelity *= apply_syndrome_operator(mps, Urot)
    measure_ancilla(mps, is_postselect=True)

    if init_state == 1:
        apply_Xlogical(mps)
    
    fidelity *= mps.canonicalization(threshold=1-0.1**threshold_err)
    if is_print:
        print(f"virtual dims: {mps.virtual_dims}")
    
    mps.apex = 0

    syndrome = []
    virtual_dim_shot = []

    for round in range(nround-1):
        if is_print or is_print_large:
            print(f"------ round {round} ------")

        # apply amplitude damping channel error for each data and ancilla
        apply_kraus(mps, kraus_list=kraus_list)

        # execute syndrome measurement
        fidelity *= apply_syndrome_operator(mps, Urot)
        if is_print:
            print(f"virtual dims: {mps.virtual_dims}")
        
        # measure ancilla
        syndrome_round, prob_list_round = measure_ancilla(mps, is_postselect=False)
        if is_print:
            print(f"virtual dims: {mps.virtual_dims}")
        
        fidelity *= mps.canonicalization(threshold=1-0.1**threshold_err)
        virtual_dim_shot.append(mps.virtual_dims)
        syndrome.append(syndrome_round)
        if is_print or is_print_large:
            print(f"virtual dims after truncation: {mps.virtual_dims}")
            print(f"fidelity: {fidelity}")
    
    # apply H to all data qubit
    for q in range(qnum):
        mps.apply_MPO([q], MPO(mpoH.tensors))

    # final measurement at data qubit
    syndrome_final, prob_list_final = measure_data(mps, is_postselect=False)

    if is_print:
        print(f"final mpo virtual dims:", mps.virtual_dims)

    # convert final measurement at data qubit to final x-syndrome
    x_syndrome_final = []
    for col in range((ncolumn-1)//2):
        pos = col * 6
        x_stab = [[2, 5], [0, 1, 3, 4], [4, 5, 7, 8], [3, 6]]
        for xs in x_stab:
            sval = 0
            for q in xs:
                sval ^= syndrome_final[pos+q]
            x_syndrome_final.append(sval)

    x_syndrome_idx = []
    for col in range((ncolumn-1)//2):
        x_syndrome_idx += [col*6+1,col*6+3,col*6+4,col*6+6]
    x_syndrome = np.array(syndrome)[:,x_syndrome_idx]
    x_syndrome = np.row_stack([x_syndrome, x_syndrome_final])
    
    if is_print:
        print(f"x-syndrome measurement result:")
        for round in range(nround):
            print(f"round {round}: {x_syndrome[round]}")
    
    # calculate difference syndrome
    x_syndrome[1:] = (x_syndrome[1:] - x_syndrome[0:-1]) % 2

    if is_print:
        print(f"xor syndrome measurement result:")
        for round in range(nround):
            print(f"round {round}: {x_syndrome[round]}")

    # old and wrong
    # Hz = np.array([[0,0,1,0,0,1,0,0,0],[1,1,0,1,1,0,0,0,0],[0,0,0,1,0,0,1,0,0],[0,0,0,0,1,1,0,1,1]])
    Hz = []
    for col in range((ncolumn-1)//2):
        pos = col * 6
        x_stab = [[2, 5], [0, 1, 3, 4], [4, 5, 7, 8], [3, 6]]
        for xs in x_stab:
            xlist = [0 for _ in range(3 * ncolumn)]
            for xi in xs:
                xlist[pos+xi] = 1
            Hz.append(xlist)
    
    Hz = np.array(Hz)
    #print(Hz)

    p = 0.01
    q = 0.01
    m = Matching(Hz, spacelike_weights=np.log((1-p)/p), repetitions=nround, timelike_weights=np.log((1-q)/q))

    x_correction_syndrome = m.decode(x_syndrome.T)

    corrected_syndrome_final = np.logical_xor(syndrome_final, x_correction_syndrome).astype(int)
    xlogical_result = np.sum(corrected_syndrome_final) % 2

    # classical logical error rate
    cler = xlogical_result if init_state == 0 else 1 - xlogical_result

    # averaged bond dim
    bond_dim_ave = np.average(np.array(virtual_dim_shot).flatten())

    if is_print:
        print(f"final measurement outcome on data qubit")
        print(np.array(syndrome_final).reshape(-1, 3).T)
        print("x-correction syndrome:")
        print(np.array(x_correction_syndrome).reshape(-1, 3).T)
        print("corrected final syndrome:")
        print(np.array(corrected_syndrome_final).reshape(-1, 3).T)
        print("corrected x logical result:", xlogical_result)
        print("classcal ler:", cler) 
        print(f"bond dim ave: {bond_dim_ave}")
        print(f"fidelity: {fidelity}")

    return cler, bond_dim_ave, fidelity

classical_logical_error_list = []
elapsed_time_list = []
bond_dim_ave_list = []
fidelity_list = []

for shot in tqdm.tqdm(range(nshot)):
    #print(f"{shot}th shot")
    start = time.time()
    
    cler, bond_dim_ave, fidelity = execute()
    classical_logical_error_list.append(cler)
    bond_dim_ave_list.append(bond_dim_ave)
    fidelity_list.append(fidelity)
    end = time.time()
    elapsed_time_list.append(end - start)
    if is_print_large or is_print:
        print(f"elapsed time for epoch {shot}: {end - start}[s]")

    if shot % print_shot == print_shot-1 or shot == nshot-1:
        cler_ave, cler_std = np.mean(np.array(classical_logical_error_list)), np.std(np.array(classical_logical_error_list)) / np.sqrt(len(classical_logical_error_list))
        print(f"clerclassical logical error rate ave: {cler_ave}±{cler_std}")
        cler_ave, cler_std = np.mean(np.array(classical_logical_error_list[-100:])), np.std(np.array(classical_logical_error_list[-100:])) / np.sqrt(100)
        print(f"recent clerclassical logical error rate ave: {cler_ave}±{cler_std}")
        bdim_ave, bdim_std = np.mean(np.array(bond_dim_ave_list)), np.std(np.array(bond_dim_ave_list)) / np.sqrt(len(bond_dim_ave_list))
        print(f"bond dim ave: {bdim_ave}±{bdim_std}")
        fid_ave, fid_std = np.mean(np.array(fidelity_list)), np.std(np.array(fidelity_list)) / np.sqrt(len(fidelity_list))
        print(f"fidelity ave: {fid_ave}±{fid_std:.3f}")
        time_ave, time_std = np.mean(np.array(elapsed_time_list)), np.std(np.array(elapsed_time_list)) / np.sqrt(len(elapsed_time_list))
        print(f"elapsed time ave: {time_ave:.2f}±{time_std:.3f}[s]")


if save_folder != "None" and save_folder != "False":
    os.makedirs(f"{save_folder}/data/ncolumn{ncolumn}/nround{nround}/seed{seed}", exist_ok=True)
    np.save(f"{save_folder}/data/ncolumn{ncolumn}/nround{nround}/seed{seed}/theta{theta}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_classical_logical_error_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.array(classical_logical_error_list))
    np.save(f"{save_folder}/data/ncolumn{ncolumn}/nround{nround}/seed{seed}/theta{theta}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_elapsed_time_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.array(elapsed_time_list))
    np.save(f"{save_folder}/data/ncolumn{ncolumn}/nround{nround}/seed{seed}/theta{theta}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_bond_dim_ave_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.array(bond_dim_ave_list))
    np.save(f"{save_folder}/data/ncolumn{ncolumn}/nround{nround}/seed{seed}/theta{theta}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_fidelity_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.array(fidelity_list))