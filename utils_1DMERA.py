import numpy as np
from scipy import integrate
import tensorflow as tf  # tf 2.x
import tensornetwork as tn
import cotengra as ctg
import QGOpt as qgo
from tqdm import tqdm
import os
import opt_einsum as oe
from tn_qsim.mera3D import MERA3D
tn.set_default_backend("tensorflow")
import sys
import time
import functools


def mera_layer_func(H, U, V, tdim_list, renormalize_func, l):
    """calculate hamiltonian renormalization at lth layer using H, U, V, W
    
    Args:
        H (tf.Variable): hamiltonian to be renormalized
        U, V (tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_func (list of list of func) : renormalize function for each layer and each H position
        l (int) : target layer
    """
    # index dimension before renormalization
    tdim = tdim_list[l]

    # index dimension after renormalization
    ntdim = tdim_list[l+1]

    tensor_order = ["H", "U", "V", "V", "Uc", "Vc", "Vc"]

    tensors = []
    for o in tensor_order:
        if o == "H":
            tensors.append(H)
        elif o == "U":
            tensors.append(tf.reshape(U, (tdim, tdim, tdim, tdim)))
        elif o == "Uc":
            tensors.append(tf.reshape(tf.math.conj(U), (tdim, tdim, tdim, tdim)))
        elif o == "V":
            tensors.append(tf.reshape(V, (tdim, tdim, tdim, tdim)))
        elif o == "Vc":
            tensors.append(tf.reshape(tf.math.conj(V), (tdim, tdim, tdim, tdim)))
    
    res = renormalize_func[l][0](tensors)

    for i in range(1, len(renormalize_func[l])):
        res += renormalize_func[l][i](tensors)

    return res / len(renormalize_func[l])

def mera_layer_pos(H, U, V, W, tdim_list, renormalize_func, l, i):
    """calculate hamiltonian renormalization at lth layer, ith position of Hamiltonian, using H, U, V, W

    Args:
        H (tf.Variable): hamiltonian to be renormalized
        U, V, W (tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_func (list of list of func) : renormalize function for each layer and each H position
        l (int) : target layer
        i (int) : target position of hamiltonian
    """
    # index dimension before renormalization
    tdim = tdim_list[l]

    # index dimension after renormalization
    ntdim = tdim_list[l+1]

    tensor_order = ["H", "U", "V", "V", "V", "V", "W", "W", "W", "W", "Uc", "Vc", "Vc", "Vc", "Vc", "Wc", "Wc", "Wc", "Wc"]

    tensors = []
    for o in tensor_order:
        if o == "H":
            tensors.append(H)
        elif o == "U":
            tensors.append(tf.reshape(U, (tdim, tdim, tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "Uc":
            tensors.append(tf.reshape(tf.math.conj(U), (tdim, tdim, tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "V":
            tensors.append(tf.reshape(V, (tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "Vc":
            tensors.append(tf.reshape(tf.math.conj(V), (tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "W":
            tensors.append(tf.reshape(W, (tdim, tdim, tdim, tdim, tdim, ntdim)))
        elif o == "Wc":
            tensors.append(tf.reshape(tf.math.conj(W), (tdim, tdim, tdim, tdim, tdim, ntdim)))
    
    res = renormalize_func[l][i](tensors)

    return res / len(renormalize_func[l])

def mera_layer_slice(H, U, V, tdim_list, renormalize_tree, l, i, s):
    """calculate renormalization at lth layer, ith position of Hamiltonian, sth slicing, using H, U, V, W

    Args:
        H (tf.Variable): hamiltonian to be renormalized
        U, V (tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_tree (list of list of ctg.ContractionTree) : renormalize tree for each layer and each H position
        l (int) : target layer
        i (int) : target position of hamiltonian
        s (int) : target slicing
    """
    # index dimension before renormalization
    tdim = tdim_list[l]

    # index dimension after renormalization
    ntdim = tdim_list[l+1]

    tensor_order = ["H", "U", "V", "V", "Uc", "Vc", "Vc"]

    tensors = []
    for o in tensor_order:
        if o == "H":
            tensors.append(H)
        elif o == "U":
            tensors.append(tf.reshape(U, (tdim, tdim, tdim, tdim)))
        elif o == "Uc":
            tensors.append(tf.reshape(tf.math.conj(U), (tdim, tdim, tdim, tdim)))
        elif o == "V":
            tensors.append(tf.reshape(V, (tdim, tdim, tdim, tdim)))
        elif o == "Vc":
            tensors.append(tf.reshape(tf.math.conj(V), (tdim, tdim, tdim, tdim)))
    
    tree = renormalize_tree[l][i]
    contract_core = functools.partial(tree.contract_core, backend="tensorflow")
    return contract_core(tree.slice_arrays(tensors, s)) / len(renormalize_tree[l])

def mera_layer_grad_slice(H, rho, U, V, W, tdim_list, grad_tree, l, i, idx, s):
    """calculate grad at lth layer, ith position of Hamiltonian, idxh-th unitary/isometry, sth slicing, using H, U, V, W

    Args:
        H (tf.Variable): hamiltonian to be renormalized
        rho (tf.Variable): reduced rho
        U, V, W (tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_tree (list of list of ctg.ContractionTree) : renormalize tree for each layer and each H position
        l (int) : target layer
        i (int) : target position of hamiltonian
        idx (int) : target index of isometry/unitary
        s (int) : target slicing
    """
    # index dimension before renormalization
    tdim = tdim_list[l]

    # index dimension after renormalization
    ntdim = tdim_list[l+1]

    tensor_order = ["H", "U", "V", "V", "V", "V", "W", "W", "W", "W", "Uc", "Vc", "Vc", "Vc", "Vc", "Wc", "Wc", "Wc", "Wc"]

    tensors = [rho]
    for j, o in enumerate(tensor_order):
        if j == idx + 1: 
            continue
        if o == "H":
            tensors.append(H)
        elif o == "U":
            tensors.append(tf.reshape(U, (tdim, tdim, tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "Uc":
            tensors.append(tf.reshape(tf.math.conj(U), (tdim, tdim, tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "V":
            tensors.append(tf.reshape(V, (tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "Vc":
            tensors.append(tf.reshape(tf.math.conj(V), (tdim, tdim, tdim, tdim, tdim, tdim)))
        elif o == "W":
            tensors.append(tf.reshape(W, (tdim, tdim, tdim, tdim, tdim, ntdim)))
        elif o == "Wc":
            tensors.append(tf.reshape(tf.math.conj(W), (tdim, tdim, tdim, tdim, tdim, ntdim)))
    
    tree = grad_tree[l][i][idx]
    contract_core = functools.partial(tree.contract_core, backend="tensorflow")
    return contract_core(tree.slice_arrays(tensors, s)) / len(grad_tree[l])

def inv_mera_layer_func(rho, U, V, tdim_list, inv_renormalize_func, l):
    """calculate invert density-op renormalization at lth layer using H, U, V, W

    Args:
        rho (tf.Variable): density op to be renormalized
        U, V, W (tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        inv_renormalize_func (list of list of func) : inv-renormalize function for each layer and each H position
        l (int) : target layer
    """
    # index dimension before renormalization
    tdim = tdim_list[l]

    # index dimension after renormalization
    ntdim = tdim_list[l+1]

    tensor_order = ["rho", "U", "V", "V", "Uc", "Vc", "Vc"]

    tensors = []
    for o in tensor_order:
        if o == "rho":
            tensors.append(rho)
        elif o == "U":
            tensors.append(tf.reshape(U, (tdim, tdim, tdim, tdim)))
        elif o == "Uc":
            tensors.append(tf.reshape(tf.math.conj(U), (tdim, tdim, tdim, tdim)))
        elif o == "V":
            tensors.append(tf.reshape(V, (tdim, tdim, tdim, tdim)))
        elif o == "Vc":
            tensors.append(tf.reshape(tf.math.conj(V), (tdim, tdim, tdim, tdim)))

    res = inv_renormalize_func[l][0](tensors).numpy()

    for i in range(1, len(inv_renormalize_func[l])):
        #print(f"pos: {i}")
        res += inv_renormalize_func[l][i](tensors).numpy()

    return res / len(inv_renormalize_func[l])

def calc_energy_from_hamiltonian(hamiltonian, U_var, V_var, psi_var, tdim_list, renormalize_func):
    """calculate energy for hamiltonian

    Args:
        hamiltonian (tf.Variable): target hamiltonian
        U_var, V_var, psi_var (list of tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_func (list of list of func) : renormalize function for each layer and each H position
    """
    # convert real valued variables back to complex valued tensors
    U_var_c = list(map(qgo.manifolds.real_to_complex, U_var))
    V_var_c = list(map(qgo.manifolds.real_to_complex, V_var))
    psi_var_c = qgo.manifolds.real_to_complex(psi_var)

    # initial local Hamiltonian term
    h_renorm = hamiltonian

    # renormalization of a local Hamiltonian term
    for l in range(len(U_var)):
        h_renorm = mera_layer_func(h_renorm, U_var_c[l], V_var_c[l], tdim_list, renormalize_func, l)

    # renormalizad Hamiltonian (low dimensional)
    h_renorm = (h_renorm + tf.transpose(h_renorm, (1, 0, 3, 2))) / 2
    h_renorm = tf.reshape(h_renorm, (tdim_list[-1] ** 2, tdim_list[-1] ** 2))

    # energy
    E = tf.cast((tf.linalg.adjoint(psi_var_c) @ h_renorm @ psi_var_c), dtype=tf.float64)[0, 0]
    
    return E

def calc_energy_func(hamiltonian, reduced_rho, U_var, V_var, tdim_list, renormalize_func, l):
    """calculate energy using renormalized hamiltonian and reduced rho at lth layer
    
    Args:
        hamiltonian (tf.Variable) : renormalized hamiltonian at layer l
        reduced_rho (tf.Variable) : inv-renormalized density op at layer l+1
        U_var, V_var (list of tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_func (list of list of func) : renormalize function for each layer and each H position
        l (int) : target layer
    """
    # convert real valued variables back to complex valued tensors
    U_var_c = qgo.manifolds.real_to_complex(U_var)
    V_var_c = qgo.manifolds.real_to_complex(V_var)

    # initial local Hamiltonian term
    h_renorm = hamiltonian

    # renormalization of a local Hamiltonian term
    h_renorm = mera_layer_func(h_renorm, U_var_c, V_var_c, tdim_list, renormalize_func, l)

    # renormalizad Hamiltonian (low dimensional)
    # h_renorm = (h_renorm + tf.transpose(h_renorm, (3, 2, 1, 0, 7, 6, 5, 4))) / 2

    # energy
    E = oe.contract("abcd,abcd", h_renorm, reduced_rho, backend="tensorflow")
    
    return h_renorm, E

def calc_energy_slice(hamiltonian, reduced_rho, U_var, V_var, tdim_list, renormalize_tree, l, i, s):
    """calculate energy using renormalized hamiltonian and reduced rho at lth layer, ith position and sth slicing

    Args:
        hamiltonian (tf.Variable) : renormalized hamiltonian at layer l
        reduced_rho (tf.Variable) : inv-renormalized density op at layer l
        U_var, V_var (list of tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_tree (list of list of ctg.ContractionTree) : renormalize tree for each layer and each H position
        l (int) : target layer
        i (int) : target position
        s (int) : target slicing index
    """
    # convert real valued variables back to complex valued tensors
    U_var_c = qgo.manifolds.real_to_complex(U_var)
    V_var_c = qgo.manifolds.real_to_complex(V_var)

    # initial local Hamiltonian term
    h_renorm = hamiltonian

    # renormalization of a local Hamiltonian term
    h_renorm = mera_layer_slice(h_renorm, U_var_c, V_var_c, tdim_list, renormalize_tree, l, i, s)

    # renormalizad Hamiltonian (low dimensional)
    # h_renorm = (h_renorm + tf.transpose(h_renorm, (3, 2, 1, 0, 7, 6, 5, 4))) / 2

    # energy
    E = oe.contract("abcd,abcd", h_renorm, reduced_rho, backend="tensorflow")
    
    return h_renorm, E

def calc_grad_slice(hamiltonian, reduced_rho, U_var, V_var, W_var, tdim_list, grad_tree, l, i, idx, s):
    """calculate energy using renormalized hamiltonian and reduced rho at lth layer, ith position and sth slicing

    Args:
        hamiltonian (tf.Variable) : renormalized hamiltonian at layer l
        reduced_rho (tf.Variable) : inv-renormalized density op at layer l
        U_var, V_var, W_var (list of tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        renormalize_tree (list of list of ctg.ContractionTree) : renormalize tree for each layer and each H position
        l (int) : target layer
        i (int) : target position
        idx (int) : index of isometry/unitary
        s (int) : target slicing index
    """
    # convert real valued variables back to complex valued tensors
    U_var_c = qgo.manifolds.real_to_complex(U_var)
    V_var_c = qgo.manifolds.real_to_complex(V_var)
    W_var_c = qgo.manifolds.real_to_complex(W_var)

    # initial local Hamiltonian term
    h_renorm = hamiltonian

    grad = mera_layer_grad_slice(h_renorm, reduced_rho, U_var_c, V_var_c, W_var_c, tdim_list, grad_tree, l, i, idx, s)
    
    return grad

def calc_reduced_rho(U_var, V_var, psi_var, tdim_list, inv_renormalize_func):
    """return reduced_rho list for all layers
    
    Args:
        U_var, V_var, psi_var (list of tf.Variable) : mera layer tensors
        tdim_list (list of int) : index dimension for each layer
        inv_renormalize_func (list of list of func) : inv-renormalize function for each layer and each H position
    """
    # convert real valued variables back to complex valued tensors
    U_var_c = list(map(qgo.manifolds.real_to_complex, U_var))
    V_var_c = list(map(qgo.manifolds.real_to_complex, V_var))
    psi_var_c = qgo.manifolds.real_to_complex(psi_var)
    
    bdim = tdim_list[-1]
    psi_renorm = tf.reshape(psi_var_c, (bdim, bdim))
    rho_renorm = oe.contract("ab,ef->abef",tf.math.conj(psi_renorm), psi_renorm, backend="tensorflow")
    rho_renorm = (rho_renorm + tf.transpose(rho_renorm, (1, 0, 3, 2))) / 2.0

    reduced_rho_list = [rho_renorm]

    nlayer = len(U_var)

    for l in range(nlayer-1, -1, -1):
        #print(f"layer:{i}")
        rho_renorm = inv_mera_layer_func(rho_renorm, U_var_c[l], V_var_c[l], tdim_list, inv_renormalize_func, l)
        #rho_renorm = (rho_renorm + tf.transpose(rho_renorm, (3, 2, 1, 0, 7, 6, 5, 4))) / 2
        reduced_rho_list.append(rho_renorm)
    
    reduced_rho_list = reduced_rho_list[::-1]
    
    return reduced_rho_list
