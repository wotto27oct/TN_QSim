import numpy as np
import opt_einsum as oe
import cotengra as ctg
from tn_qsim.mpo import MPO
from tn_qsim.pepo import PEPO
from tn_qsim.pepdo import PEPDO
import copy
from scipy.linalg import expm
import sys
import time

args = sys.argv

np.set_printoptions(linewidth=500)

# config
width = int(args[1]) # only 3*3 or 5*5
height = int(args[2])
qnum = width * height
noise_type = args[3] # AD or DP or SR
gamma, eps, theta = None, None, None
if noise_type == "AD":
    gamma = float(args[4])
elif noise_type == "DP":
    eps = float(args[4])
elif noise_type == "SR":
    theta = float(args[4])
seed = int(args[5])
trial = int(args[6]) # repetation number
data_folder = args[7] # no meaning

algorithm = ctg.ReusableHyperOptimizer(
    methods="kahypar",
    max_repeats=1000,
    max_time=60,
    slicing_opts={"target_size": 2**40},
    slicing_reconf_opts={"target_size": 2**28},
    reconf_opts={"subtree_size": 14},
    progbar=True,
    directory="ctg_path_cache",
)

np.random.seed(seed)
start = time.time()

xstab_list, zstab_list, logical_cnot, recover_xstab_list, recover_zstab_list, logical_list = None, None, None, None, None, None
if width == 3 and height == 3:
    xstab_list = [[0,3,4,1],[2,5],[3,6],[7,4,5,8]]
    zstab_list = [[0,1],[1,4,5,2],[6,3,4,7],[7,8]]
    logical_cnot = [6,7,8]
    recover_xstab_list = [[0],[2],[3,0],[4,1]]
    recover_zstab_list = [[0],[4,3],[6],[7,6]]
    logical_list = [[0,1,2],[0,3,6]]
elif width == 5 and height == 5:
    xstab_list = [[0,5,6,1],[2,7,8,3],[4,9],[10,15,16,11],[12,17,18,13],[14,19],[10,5],[11,6,7,12],[13,8,9,14],[20,15],[21,16,17,22],[23,18,19,24]]
    zstab_list = [[0,1],[5,6,11,10],[15,16,21,20],[2,3],[7,8,13,12],[17,18,23,22],[2,1,6,7],[12,11,16,17],[22,21],[4,3,8,9],[14,13,18,19],[23,24]]
    logical_cnot = [20,21,22,23,24]
    recover_xstab_list = [[0],[2],[4],[10,5,0],[12,7,2],[14,9,4],[5,0],[6,1],[8,3],[15,10,5,0],[16,11,6,1],[18,13,8,3]]
    recover_zstab_list = [[0],[10],[20],[2,1,0],[12,11,10],[22,21,20],[6,5],[16,15],[21,20],[8,7,6,5],[18,17,16,15],[23,22,21,20]]
    logical_list = [[0,1,2,3,4],[0,5,10,15,20]]


stab_list = []
for xstab in xstab_list:
    stab_list.append((xstab, 0))
for zstab in zstab_list:
    stab_list.append((zstab, 1))

recover_list = []
for rxstab in recover_xstab_list:
    recover_list.append((rxstab, 0))
for rzstab in recover_zstab_list:
    recover_list.append((rzstab, 1))

zero = np.array([1, 0])
one = np.array([0, 1])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

## step1
print("------ initializing pepdo ------")
pepdo = PEPDO([zero.reshape(2,1,1,1,1,1) for i in range(qnum)], width, height)

## step2
print("------ project onto codespace ------")
def create_stab(length, C, is_minus=False):
    tensor_list = []
    for l in range(length):
        if l == 0:
            Q = np.zeros([2,2,2])
            Q[0] = np.eye(2)
            if is_minus:
                Q[1] = -C
            else:
                Q[1] = C
            Q = Q.transpose(1,2,0)
            Q = Q.reshape(2,2,1,2)
        elif l == length - 1:
            Q = np.zeros([2,2,2])
            Q[0] = np.eye(2)
            Q[1] = C
            Q = Q.transpose(1,2,0)
            Q = Q.reshape(2,2,2,1)
        else:
            Q = np.zeros([2,2,2,2])
            Q[0][0] = np.eye(2)
            Q[1][1] = C
            Q = Q.transpose(2,3,0,1)
        tensor_list.append(Q)
    mpo = MPO(tensor_list)
    return mpo

for xstab in xstab_list:
    mpo = create_stab(len(xstab), X)
    pepdo.apply_MPO(xstab, mpo)
    pepdo.nodes[0].tensor = pepdo.nodes[0].tensor / np.sqrt(pepdo.calc_trace())

## step3
print("------ apply logical CNOT ------")
mpo = create_stab(len(logical_cnot),X)
mpo.nodes[-1].tensor = mpo.nodes[-2].tensor
pepdo.apply_MPO(logical_cnot, mpo)
pepdo.nodes[logical_cnot[-1]].tensor = pepdo.nodes[logical_cnot[-1]].tensor.transpose(0,1,5,3,4,2)

print(pepdo.vertical_virtual_dims)
print(pepdo.horizontal_virtual_dims)
print(pepdo.inner_dims)


## step4
print("------ apply noise ------")
## amplitude damping
if noise_type == "AD":
    K0 = oe.contract("a,b->ab",zero,zero.conj()) + np.sqrt(1-gamma) * oe.contract("a,b->ab",one,one.conj())
    K1 = np.sqrt(gamma) * oe.contract("a,b->ab",zero,one.conj())
    Ead = np.array([K0,K1]).transpose(1,2,0).reshape(2,2,1,2)
    Ead_mpo = MPO([Ead])
    for q in range(qnum):
        pepdo.apply_MPO([q], Ead_mpo)
    print(pepdo.vertical_virtual_dims)
    print(pepdo.horizontal_virtual_dims)
    print(pepdo.inner_dims)

## depolarizing channel
if noise_type == "DP":
    Edep = np.array([np.sqrt(1-eps)*np.eye(2), np.sqrt(eps)*X, np.sqrt(eps)*Y, np.sqrt(eps)*Z]).transpose(1,2,0).reshape(2,2,1,4)
    print(Edep.shape)
    Edep_mpo = MPO([Edep])
    for q in range(qnum):
        print("trace before apply_MPO:", pepdo.calc_trace())
        pepdo.apply_MPO([q], Edep_mpo)
        print("trace after apply_MPO:", pepdo.calc_trace())
    print(pepdo.vertical_virtual_dims)
    print(pepdo.horizontal_virtual_dims)
    print(pepdo.inner_dims)

## symmetric rotation
if noise_type == "SR":
    Esr = expm(-1j*theta*Z)
    Esr_mpo = MPO([Esr.reshape(2,2,1,1)])
    for q in range(qnum):
        print("trace before apply_MPO:", pepdo.calc_trace())
        pepdo.apply_MPO([q], Esr_mpo)
        print("trace after apply_MPO:", pepdo.calc_trace())
    print(pepdo.vertical_virtual_dims)
    print(pepdo.horizontal_virtual_dims)
    print(pepdo.inner_dims)

## step5
print("------ syndrome measurement ------")

syndrome_list = []
prob_list = []
minimum_distance_list = []
pepdo_tensor = pepdo.tensors

for error in range(trial):
    print(f"trial: {error}")
    syndrome = []
    prob = 1.0
    pepdo = PEPDO(pepdo_tensor, height, width)
    for idx, (stab, type) in enumerate(stab_list):
        stab_type = "x"
        W = X
        if type == 1:
            stab_type = "z"
            W = Z
        print(f"{stab_type}-stab for {stab}")
        
        pepdo_tmp = PEPDO(copy.deepcopy(pepdo.tensors), height, width)
        mpo = create_stab(len(stab), W)
        pepdo_tmp.apply_MPO(stab, mpo)
        tree, cost, sp_cost = pepdo_tmp.find_trace_tree(algorithm)
        print(f"calc pr0. slice: {tree.sliced_inds} total_flops: {tree.total_flops():,} cost: {cost:,} sp_cost: {sp_cost:,}")
        pr0 = np.real_if_close(pepdo_tmp.calc_trace(tree=tree)[0][0])

        pepdo_tmp = PEPDO(copy.deepcopy(pepdo.tensors), height, width)
        mpo = create_stab(len(stab), W, is_minus=True)
        pepdo_tmp.apply_MPO(stab, mpo)
        tree, cost, sp_cost = pepdo_tmp.find_trace_tree(algorithm)
        print(f"calc pr1. slice: {tree.sliced_inds} total_flops: {tree.total_flops():,} cost: {cost:,} sp_cost: {sp_cost:,}")
        pr1 = np.real_if_close(pepdo_tmp.calc_trace(tree=tree)[0][0])
        
        pr0, pr1 = pr0 / (pr0 + pr1), pr1 / (pr0 + pr1)
        print(f"syndrome pr0={pr0}, pr1={pr1}")
        prob_random_value = np.random.uniform()
        #prob_random_value = error & (1 << idx)
        if pr0 > prob_random_value:
            print("0 selected")
            syndrome.append(0)
            mpo = create_stab(len(stab), W)
            pepdo.apply_MPO(stab, mpo)
            prob *= pr0
        else:
            print("1 selected")
            syndrome.append(1)
            mpo = create_stab(len(stab), W, is_minus=True)
            pepdo.apply_MPO(stab, mpo)
            prob *= pr1
        tree, cost, sp_cost = pepdo.find_trace_tree(algorithm)
        print(f"calc trace. slice: {tree.sliced_inds} total_flops: {tree.total_flops():,} cost: {cost:,} sp_cost: {sp_cost:,}")
        trace = np.sqrt(pepdo.calc_trace(tree=tree)[0][0])
        for node in pepdo.nodes:
            node.tensor = node.tensor / np.power(trace, 1/qnum)

        print(pepdo.vertical_virtual_dims)
        print(pepdo.horizontal_virtual_dims)
        print(pepdo.inner_dims)

    """print(pepdo.calc_trace())
    sub_trace = np.trace(pepdo.calc_trace())
    for node in pepdo.nodes:
        node.tensor = node.tensor / np.power(np.sqrt(sub_trace), 1/qnum)
    print(pepdo.calc_trace())
    print(np.trace(pepdo.calc_trace()))"""

    print("syndrome", syndrome)
    prob_list.append(prob)

    # step6
    print("------ apply first error-correction pauli ------")
    for idx, syn in enumerate(syndrome):
        if syn == 1:
            recover_stab_type = "x"
            W = Z
            if recover_list[idx][1]:
                recover_stab_type = "z"
                W = X
            print(f"{recover_stab_type}-stab for {recover_list[idx][0]}")
            for recover in recover_list[idx][0]:
                mpo = MPO([W.reshape(2,2,1,1)])
                pepdo.apply_MPO([recover], mpo)

    
    print(pepdo.vertical_virtual_dims)
    print(pepdo.horizontal_virtual_dims)
    print(pepdo.inner_dims)

    #calc Ai, branching
    # step7
    print("------ calc QEC channel ------")
    Ai_pepdo_tensor = pepdo.tensors
    Ai_list = []
    Ai0_list = []
    for i in range(4):
        print("i:", i)
        print("apply Pi")
        pepdo = PEPDO(Ai_pepdo_tensor, width, height)
        if i == 1 or i == 2:
            # logical X
            for idx in logical_list[0]:
                mpo = MPO([X.reshape(2,2,1,1)])
                pepdo.apply_MPO([idx], mpo)
        if i // 2 == 1:
            # logical Z
            for idx in logical_list[1]:
                mpo = MPO([Z.reshape(2,2,1,1)])
                pepdo.apply_MPO([idx], mpo)
        
        print("apply second error-correction pauli")
        for idx, syn in enumerate(syndrome):
            if syn == 1:
                recover_stab_type = "x"
                W = Z
                if recover_list[idx][1]:
                    recover_stab_type = "z"
                    W = X
                print(f"{recover_stab_type}-stab for {recover_list[idx][0]}")
                for recover in recover_list[idx][0]:
                    mpo = MPO([W.reshape(2,2,1,1)])
                    pepdo.apply_MPO([recover], mpo)
                
        #print(np.trace(pepdo.calc_trace()))
        
        #Ai = pepdo.calc_trace()
        pepo1 = PEPO([tensor.transpose(0,5,1,2,3,4) for tensor in pepdo.tensors], height, width)
        pepo2 = PEPO([tensor.transpose(0,5,1,2,3,4).conj() for tensor in pepdo_tensor], height, width)
        
        tree, cost, sp_cost = pepo1.find_pepo_trace_tree(pepo2, algorithm)
        print(f"calc Ai. slice: {tree.sliced_inds} total_flops: {tree.total_flops():,} cost: {cost:,} sp_cost: {sp_cost:,}")
        Ai = pepo1.calc_pepo_trace(pepo2, tree=tree) / 2.0
        Ai_list.append(Ai)
        Ai0_list.append(Ai[0][0])
    
    C = np.zeros([4,4], dtype="complex128")
    for i in range(4):
        C[i][0] = np.trace(Ai_list[i]) / 2.0
        C[i][1] = np.trace(np.dot(X, Ai_list[i])) / 2.0
        C[i][2] = -1j * np.trace(np.dot(Y, Ai_list[i])) / 2.0
        C[i][3] = np.trace(np.dot(Z, Ai_list[i])) / 2.0

    Cmat = np.zeros([4,4], dtype="complex128")
    Pauli = [np.eye(2), X, Y, Z]
    for i in range(4):
        for j in range(4):
            Cmat += C[i][j] * oe.contract("ab,cd->abcd",Pauli[i],Pauli[j]).reshape(4,4)

    moto_Cmat = Cmat.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    #print("moto", moto_Cmat)
    trace = np.trace(moto_Cmat)
    #print("moto_trace", trace)
    #print(Cmat)
    # regularization, due to Choi-jamiolkowski
    Cmat = Cmat * 2 / trace
    print("logical map C:")
    print(Cmat)
    Cmat = Cmat.reshape(2,2,2,2)
    
    # step8
    print("------ calc optimal decoding ------")
    minumum_distance = 0
    for i in range(4):
        distance = np.linalg.norm(oe.contract("abcd,Aa,Bb->ABcd",Cmat,Pauli[i],Pauli[i].conj()).reshape(4,4) - np.eye(4))
        print(i, "distance", distance)
        if i == 0 or distance < minumum_distance:
            minumum_distance = distance
    print("minimum distance:", minumum_distance)
    minimum_distance_list.append(minumum_distance)

ave_minimum_distance = np.average(np.array(minimum_distance_list))
print(f"average logical error rate: {ave_minimum_distance}")

end = time.time()
print(f"elapsed time: {end - start}[s]")