import opt_einsum as oe
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import unitary_group

from tn_qsim.mps import MPS
from tn_qsim.mpo import MPO

import numpy as np
import opt_einsum as oe
from tn_qsim.mps import MPS
from tn_qsim.mpo import MPO
from tn_qsim.peps import PEPS
from tn_qsim.peps3D import PEPS3D
from tn_qsim.tn3D import TN3D
from scipy.stats import unitary_group, ortho_group
import time
import functools

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(linewidth=500)


def test_contract_init_state():
    zero = np.array([1, 0])
    one = np.array([0, 1])
    tensors = [zero, one, one, zero, zero, one, one, zero, zero]
    state = oe.contract("a,b,c,d,e,f,g,h,i->abcdefghi",*tensors).flatten()

    tn3D = TN3D(tensors, 9)
    state_tn3D = tn3D.amplitude([None for _ in range(9)], algorithm="greedy").flatten()

    assert_allclose(state, state_tn3D)

def test_apply_single_MPO():
    zero = np.array([1, 0])
    one = np.array([0, 1])
    tensors = [zero, zero, zero, zero, zero, zero, zero, zero, zero]

    tn3D = TN3D(tensors, 9)
    
    X = np.array([[0, 1], [1, 0]])
    mpo = MPO([X.reshape(2,2,1,1)])
    tn3D.apply_MPO([0], mpo)
    state_tn3D = tn3D.amplitude([None for _ in range(9)], algorithm="greedy").flatten()

    tensors[0] = one
    state = oe.contract("a,b,c,d,e,f,g,h,i->abcdefghi",*tensors).flatten()
    assert_allclose(state, state_tn3D)

def test_apply_2_qubit_MPO():
    zero = np.array([1, 0])
    one = np.array([0, 1])
    tensors = [zero, zero, zero, zero, zero, zero, zero, zero, zero]

    tn3D = TN3D(tensors, 9)

    X = np.array([[0, 1], [1, 0]])
    mpo2 = MPO([X.reshape(2,2,1,1), X.reshape(2,2,1,1)])
    tn3D.apply_MPO([2, 5], mpo2)
    state_tn3D = tn3D.amplitude([None for _ in range(9)], algorithm="greedy").flatten()

    tensors[2] = one
    tensors[5] = one
    state = oe.contract("a,b,c,d,e,f,g,h,i->abcdefghi",*tensors).flatten()
    assert_allclose(state, state_tn3D)

def test_apply_3_qubit_MPO():
    zero = np.array([1, 0]).astype(np.complex128)
    one = np.array([0, 1]).astype(np.complex128)
    tensors4 = [zero, zero, zero, one]
    U = unitary_group.rvs(8).astype(np.complex128)
    U = U.reshape(2,2,2,2,2,2).transpose(0,3,1,4,2,5)
    U1, s1, Vh1 = np.linalg.svd(U.reshape(4,-1), full_matrices=False)
    U1 = np.dot(U1, np.diag(s1)).reshape(2,2,1,4)
    U2, s2, Vh2 = np.linalg.svd(Vh1.reshape(16, -1), full_matrices=False)
    U2 = np.dot(U2, np.diag(s2)).reshape(4,2,2,4).transpose(1,2,0,3)
    U3 = Vh2.reshape(4,2,2,1).transpose(1,2,0,3)
    print(U1.dtype)
    mpo = MPO([U1, U2, U3])

    tn3D = TN3D(tensors4, 4)
    tn3D.apply_MPO([0,1,3], mpo)
    state_tn3D = tn3D.amplitude([None for _ in range(4)], algorithm="greedy").flatten()

    state = np.einsum("i,j,k,l->ijkl",zero,zero,zero,one)
    state = oe.contract("ijkl,piqjrl->pqkr", state, U).flatten().astype(np.complex128)
    assert_allclose(state, state_tn3D)