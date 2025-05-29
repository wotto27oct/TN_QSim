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
settings = ["dqnum", "nround", "theta", "gamma", "tau", "temp", "spread", "rtype", "init_state", "seed", "nshot", "threshold_err", "save_folder"]
ns  = locals()
print("settings")
for i, arg in enumerate(args):
    if i == 0:
        continue
    ns[settings[i-1]] = args[i]
    print(f"{settings[i-1]}: {args[i]}")

########################
dqnum = int(dqnum)
nround = int(nround)
theta = float(theta) # rotation noise strength, [0, np.pi]
gamma = float(gamma) # coupling strength to bath, typically 0.1 [MHz]
tau = float(tau) # the time of the thermalization process, typically 1.0 [us]
temp = float(temp) # temperature. typically 10[mK] ~ 100[mK]
spread = float(spread) # leakage spread.
rtype = rtype # removal type, "noreset", "mlr", "dqlr"
init_state = int(init_state) # 0 or 1
seed = int(seed)
nshot = int(nshot)
threshold_err = int(threshold_err)
save_folder = save_folder
########################

stype = "mps"

if save_folder != "None" and save_folder != "False":
    if os.path.exists(f"{save_folder}/data/dqnum{dqnum}/nround{nround}/seed{seed}/theta{theta:.3f}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_classical_logical_error_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy"):
        print("result exists.")
        exit()

is_print = False
is_print_large = False
print_shot = 100

np.random.seed(seed)

qnum = 2*dqnum-1

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
R_spread_mpo = return_mpo_from_2qubit_op(R_spread, dim1=pdim, dim2=pdim)

# leakage removal
LeakageISWAP = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                    [0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1j,0,0],[0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,1j,0,0,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]).astype(np.complex128)
LeakageISWAP_mpo = return_mpo_from_2qubit_op(LeakageISWAP, dim1=pdim, dim2=pdim)

H = np.array([[1/np.sqrt(2), 1/np.sqrt(2), 0], [1/np.sqrt(2), -1/np.sqrt(2), 0], [0, 0, 1.0]])
mpoX = MPO([X.reshape(3,3,1,1)])
mpoH = MPO([H.reshape(3,3,1,1)])

cz_phi = 0.5
CZ = np.diag(np.array([1,1,1,1,-1,np.exp(1j*cz_phi*np.pi),1,np.exp(1j*cz_phi*np.pi),1]))
CZ_mpo = return_mpo_from_2qubit_op(CZ, dim1=pdim, dim2=pdim)

def initialize_Urot():
    # fix rotation
    phi = np.ones(qnum) * np.pi / 2
    lam = np.random.uniform(low=0.0, high=2*np.pi, size=qnum)
    varphi = np.random.uniform(low=0.0, high=2*np.pi, size=qnum)
    Urot = []
    for i in range(qnum):
        Urot.append(Rot(theta*np.pi, phi[i], lam[i], varphi[i]).reshape(pdim,pdim,1,1))
    return Urot

def prepare_right_tensors(mps):
    right_trace_tensors = [np.array([[1]])] # Warn! ith tensor: contract tensors to the right of the qnum-1-i th tensor
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
    right_trace_tensors = prepare_right_tensors(mps) # ith tensor: contract right qnum-1-i~qnum MPS
    left_tensor = np.array([[1]])
    for pos in range(0, qnum):
        right_tensor = right_trace_tensors[qnum-1-(pos)]
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

def apply_H_to_ancilla(mps):
    for i in range(1, qnum-1, 2):
        mps.apply_MPO([i], MPO(mpoH.tensors))

def apply_cz_and_Urot(mps, Urot, sup):
    fidelity = mps.apply_MPO(sup, MPO(CZ_mpo.tensors), is_truncate=True)
    for q in sup:
        mps.apply_MPO([q], MPO([Urot[q]]))
    return fidelity

def apply_leakage_spread(mps, sup):
    return mps.apply_MPO(sup, MPO(R_spread_mpo.tensors), is_truncate=True)

def apply_syndrome_operator(mps, Urot, is_leakage_spread=False):
    fidelity = 1.0
    for i in range(0, qnum-2, 2):
        fidelity *= apply_cz_and_Urot(mps, Urot, [i,i+1])
        if is_leakage_spread:
            fidelity *= mps.move_left_canonical(threshold=1-0.1**threshold_err)
            fidelity *= apply_leakage_spread(mps, [i,i+1])
        fidelity *= mps.move_right_canonical(threshold=1-0.1**threshold_err)
        
    for i in range(qnum-1, 1, -2):
        fidelity *= apply_cz_and_Urot(mps, Urot, [i,i-1])
        if is_leakage_spread:
            fidelity *= mps.move_right_canonical(threshold=1-0.1**threshold_err)
            fidelity *= apply_leakage_spread(mps, [i,i-1])
        fidelity *= mps.move_left_canonical(threshold=1-0.1**threshold_err)

    return fidelity

def measure_qubit(mps, pos, left_tensor, right_tensor, is_apply_mpo=True):
    pr_list = []
    left_tensor_list = []
    for i in range(pdim):
        left_tensor_list.append(oe.contract("bB,abc,ABC,aA->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj(),Proj[i].reshape(3,3)))
        pr_list.append(oe.contract("cC,cC",left_tensor_list[-1],right_tensor))
    pr_list = np.real_if_close(pr_list)
    np.putmask(pr_list, np.abs(pr_list) < 1e-8, 0)
    pr_list /= np.sum(pr_list)

    if is_print:
        print(f"pos {pos} syndrome pr0={pr_list[0]}, pr1={pr_list[1]} pr2={pr_list[2]}")

    if is_apply_mpo:
        prob_random_value = np.random.uniform()
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

def measure_ancilla(mps):
    syndrome_round = []
    prob_list_round = []

    right_trace_tensors = prepare_right_tensors(mps) # ith tensor: contract right qnum-1-i~qnum MPS
    left_tensor = np.array([[1]])

    for pos in range(qnum):
        if pos % 2 == 0:
            # data qubit
            left_tensor = oe.contract("bB,abc,aBC->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj())
        else:
            # ancilla qubit
            right_tensor = right_trace_tensors[qnum-1-(pos)]
            left_tensor, pr_list, syndrome = measure_qubit(mps, pos, left_tensor, right_tensor)
            prob_list_round.append(pr_list)
            syndrome_round.append(syndrome)
    
    return syndrome_round, prob_list_round

def measure_data(mps, is_apply_mpo=True):
    syndrome_round = []
    prob_list_round = []

    right_trace_tensors = prepare_right_tensors(mps)
    left_tensor = np.array([[1]])

    for pos in range(qnum):
        if pos % 2 == 1:
            # ancilla qubit
            left_tensor = oe.contract("bB,abc,aBC->cC",left_tensor,mps.tensors[pos],mps.tensors[pos].conj())
        else:
            # data qubit
            right_tensor = right_trace_tensors[qnum-1-(pos)]
            left_tensor, pr_list, syndrome = measure_qubit(mps, pos, left_tensor, right_tensor, is_apply_mpo=is_apply_mpo)
            prob_list_round.append(pr_list)
            syndrome_round.append(syndrome)
    
    return syndrome_round, prob_list_round

def apply_leakageiswap_and_Urot(mps, Urot, sup):
    fidelity = mps.apply_MPO(sup, MPO(LeakageISWAP_mpo.tensors), is_truncate=True)
    for q in sup:
        mps.apply_MPO([q], MPO([Urot[q]]))
    return fidelity

def apply_dqlr(mps, Urot, is_downward=True):
    fidelity = 1.0
    if is_downward:
        for i in range(0, qnum-1, 2):
            fidelity *= apply_leakageiswap_and_Urot(mps, Urot, [i,i+1])
            if i < qnum-2:
                fidelity *= mps.move_right_canonical(threshold=1-0.1**threshold_err)
    else:
        for i in range(qnum-1, 1, -2):
            fidelity *= apply_leakageiswap_and_Urot(mps, Urot, [i,i-1])
            if i > 1:
                fidelity *= mps.move_left_canonical(threshold=1-0.1**threshold_err)

    # reset all measure qubit
    measure_ancilla(mps)

    return fidelity

def decode(syndrome):
    syndrome = np.array(syndrome)
    syndrome[1:] = (syndrome[1:] - syndrome[0:-1]) % 2

    if is_print:
        print(f"xor syndrome measurement result:")
        for round in range(nround+1):
            print(f"round {round}: {syndrome[round]}")

    Hz = []
    for i in range(dqnum-1):
        zstab = [0 for j in range(0, i)] + [1, 1] + [0 for j in range(i+2, dqnum)]
        Hz.append(zstab)
    Hz = np.array(Hz)
    p = 0.01
    q = 0.01
    m = Matching(Hz, spacelike_weights=np.log((1-p)/p), repetitions=nround+1, timelike_weights=np.log((1-q)/q))

    correction_syndrome = m.decode(syndrome.T)
    return correction_syndrome

def execute():
    # fix rotation
    Urot = initialize_Urot()

    # prepare MPS
    mps = MPS([basis[0].reshape(pdim,1,1) for _ in range(qnum)], threshold_err=0.1**threshold_err)

    # encode logical qubit
    if init_state == 1:
        for i in range(0, qnum, 2):
            mps.apply_MPO([i], MPO(mpoX.tensors))
    
    mps.apex = 0
    fidelity = 1.0

    syndrome = []
    virtual_dim_shot = []

    for round in range(nround):
        if is_print or is_print_large:
            print(f"------ round {round} ------")

        # apply amplitude damping channel error for each data and ancilla
        apply_kraus(mps, kraus_list=kraus_list)

        # syndrome measurement
        # first apply Hadamard to ancilla
        apply_H_to_ancilla(mps)

        # apply cz and rotation noise
        fidelity *= apply_syndrome_operator(mps, Urot, is_leakage_spread=True)

        # finally apply Hadamard to ancilla
        apply_H_to_ancilla(mps)

        # measure ancilla
        syndrome_round, prob_list_round = measure_ancilla(mps)

        # leakage removal (dqlr)
        if rtype == "dqlr":
            # apply dqlr
            apply_dqlr(mps, Urot, is_downward=(round % 2 == 0))

        if is_print:
            print(f"virtual dims: {mps.virtual_dims}")
        #mps.canonicalization(threshold=1-0.1**threshold_err)
        virtual_dim_shot.append(mps.virtual_dims)
        if is_print or is_print_large:
            print(f"virtual dims after truncation: {mps.virtual_dims}")
            print(f"fidelity: {fidelity}")

        syndrome.append(syndrome_round)
    
    # final measurement at data qubit
    syndrome_final, prob_list_final = measure_data(mps)

    if is_print:
        print(f"final mpo virtual dims:", mps.virtual_dims)

    if is_print:
        print(f"final measurement outcome on data qubit", syndrome_final)
    
    # convert final measurement at data qubit to final syndrome
    final_list = []
    for i in range(len(syndrome_final)-1):
        final_list.append(int(np.logical_xor(syndrome_final[i], syndrome_final[i+1])))
    syndrome.append(final_list)
    
    if is_print:
        print(f"syndrome measurement result:")
        for round in range(nround+1):
            print(f"round {round}: {syndrome[round]}")

    # decode 
    correction_syndrome = decode(syndrome)
    
    if is_print:
        print("correction syndrome:")
        print(correction_syndrome)

    # classical logical error rate
    corrected_syndrome_final = np.logical_xor(syndrome_final, correction_syndrome).astype(int)
    c_result, _ = stats.mode(corrected_syndrome_final, keepdims=True)
    cler = c_result[0] if init_state == 0 else 1 - c_result[0]
    if is_print:
        print("corrected syndrome:", corrected_syndrome_final, "classcal ler:", cler)

    bond_dim_ave, bond_dim_std = np.average(np.array(virtual_dim_shot).flatten()), np.std(np.array(virtual_dim_shot).flatten())

    if is_print:
        print(f"bond dim ave:{bond_dim_ave}, std:{bond_dim_std}")
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
    if is_print_large:
        print(f"elapsed time for epoch {shot}: {end - start}[s]")

    if shot % print_shot == print_shot-1 or shot == nshot-1:
        cler_ave, cler_std = np.mean(np.array(classical_logical_error_list)), np.std(np.array(classical_logical_error_list)) / np.sqrt(len(classical_logical_error_list))
        print(f"classical logical error rate ave: {cler_ave:.6f}±{cler_std:.3f}")
        cler_ave, cler_std = np.mean(np.array(classical_logical_error_list[-100:])), np.std(np.array(classical_logical_error_list[-100:])) / np.sqrt(100)
        print(f"recent clerclassical logical error rate ave: {cler_ave:.6f}±{cler_std:.3f}")
        bdim_ave, bdim_std = np.mean(np.array(bond_dim_ave_list)), np.std(np.array(bond_dim_ave_list)) / np.sqrt(len(bond_dim_ave_list))
        print(f"bond dim ave: {bdim_ave:.2f}±{bdim_std:.3f}")
        fid_ave, fid_std = np.mean(np.array(fidelity_list)), np.std(np.array(fidelity_list)) / np.sqrt(len(fidelity_list))
        print(f"fidelity ave: {fid_ave}±{fid_std:.3f}")
        time_ave, time_std = np.mean(np.array(elapsed_time_list)), np.std(np.array(elapsed_time_list)) / np.sqrt(len(elapsed_time_list))
        print(f"elapsed time ave: {time_ave:.2f}±{time_std:.3f}[s]")


if save_folder != "None" and save_folder != "False":
    os.makedirs(f"{save_folder}/data/dqnum{dqnum}/nround{nround}/seed{seed}", exist_ok=True)
    np.save(f"{save_folder}/data/dqnum{dqnum}/nround{nround}/seed{seed}/theta{theta:.3f}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_classical_logical_error_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.array(classical_logical_error_list))
    np.save(f"{save_folder}/data/dqnum{dqnum}/nround{nround}/seed{seed}/theta{theta:.3f}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_bond_dim_ave_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.mean(np.array(bond_dim_ave_list)))
    np.save(f"{save_folder}/data/dqnum{dqnum}/nround{nround}/seed{seed}/theta{theta:.3f}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_fidelity_ave_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.mean(np.array(fidelity_list)))
    np.save(f"{save_folder}/data/dqnum{dqnum}/nround{nround}/seed{seed}/theta{theta:.3f}_gamma{gamma:.3f}_tau{tau:.1f}_temp{int(temp)}_spread{spread:.3f}_{rtype}_{stype}_elapsed_time_ave_init_state{init_state}_threshold_err{threshold_err}_seed{seed}_nshot{nshot}.npy", np.mean(np.array(elapsed_time_list)))