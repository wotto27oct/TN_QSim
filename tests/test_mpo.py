import opt_einsum as oe
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import unitary_group

from tn_qsim.mps import MPS
from tn_qsim.mpo import MPO

def test_contract():
    A = np.random.randn(2,2,1,10)
    B = np.random.randn(2,2,10,20)
    C = np.random.randn(2,2,20,1)
    mpo = MPO([A, B, C])

    # contraction
    rho = mpo.contract().reshape(2**3,-1)
    rhot = oe.contract("adhi,beij,cfjk->abcdefhk",A,B,C).flatten().reshape(2**3,-1)
    assert_allclose(rho, rhot)

    # replace
    B = np.random.randn(2,2,10,20)
    mpo.replace_tensors([1], [B])
    rho = mpo.contract().reshape(2**3,-1)
    rhot = oe.contract("adhi,beij,cfjk->abcdefhk",A,B,C).flatten().reshape(2**3,-1)
    assert_allclose(rho, rhot)

def test_canonicalization():
    A = np.random.randn(2,2,1,10)
    B = np.random.randn(2,2,10,20)
    C = np.random.randn(2,2,20,1)
    mpo = MPO([A, B, C])
    orig_virtual_dims = mpo.virtual_dims
    orig_rho = mpo.contract().reshape(2**3,-1)

    # canonicalization
    mpo.canonicalization()
    virtual_dims = mpo.virtual_dims
    rho = mpo.contract().reshape(2**3,-1)
    assert np.all(virtual_dims <= orig_virtual_dims)
    assert_allclose(orig_rho, rho)

def test_apply_mpo_without_truncation():
    A = np.random.randn(2,2,1,3)
    B = np.random.randn(2,2,3,4)
    C = np.random.randn(2,2,4,1)
    mpo = MPO([A, B, C])

    D = np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5)
    F = np.random.randn(2,2,5,1)
    mpo2 = MPO([D, E, F])

    mpo.apply_MPO([0, 1, 2], mpo2)
    rho = mpo.contract().reshape(2**3,-1)
    rhot = oe.contract("abcd,efdg,higj,bklm,fnmo,ipoq->aehknp",D,E,F,A,B,C).reshape(2**3,-1)
    assert_allclose(rho, rhot)

def test_apply_mpo_inverse_direction_without_truncation():
    A = np.random.randn(2,2,1,3)
    B = np.random.randn(2,2,3,4)
    C = np.random.randn(2,2,4,1)
    mpo = MPO([A, B, C])

    D = np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5)
    mpo2 = MPO([D, E])

    mpo.apply_MPO([1,0], mpo2)
    rho = mpo.contract().reshape(-1,2**3,2**3)
    rhot = oe.contract("efgd,abdc,bhij,fkjl,mnlo->cigoaemhkn",D,E,A,B,C).reshape(-1,2**3,2**3)
    assert_allclose(rho, rhot)

def test_apply_mpo_as_CPTP_without_truncation():
    A = np.random.randn(2,2,1,3)
    B = np.random.randn(2,2,3,4)
    C = np.random.randn(2,2,4,1)
    mpo = MPO([A, B, C])

    D = np.random.randn(2,2,1,6) + 1j*np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5) + 1j*np.random.randn(2,2,6,5)
    F = np.random.randn(2,2,5,1) + 1j*np.random.randn(2,2,5,1)
    mpo2 = MPO([D, E, F])

    mpo.apply_MPO_as_CPTP([0, 1, 2], mpo2)
    rho = mpo.contract().reshape(-1,2**3,2**3)
    rhot = oe.contract("abcd,efdg,higj,bklm,fnmo,ipoq,rkst,untw,xpwy->lcsqjyaehrux",D,E,F,A,B,C,D.conj(),E.conj(),F.conj()).reshape(-1,2**3,2**3)
    assert_allclose(rho, rhot)

def test_apply_mpo_as_CPTP_inverse_direction_without_truncation():
    A = np.random.randn(2,2,1,3)
    B = np.random.randn(2,2,3,4)
    C = np.random.randn(2,2,4,1)
    mpo = MPO([A, B, C])

    D = np.random.randn(2,2,1,6) + 1j*np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5) + 1j*np.random.randn(2,2,6,5)
    mpo2 = MPO([D, E])

    mpo.apply_MPO_as_CPTP([1, 0], mpo2, is_dangling_final=True)
    rho = mpo.contract().reshape(-1,2**3,2**3)
    rhot = oe.contract("efgd,abdc,bhij,fkjl,mnlo,sktr,phrq->icqogtaempsn",D,E,A,B,C,D.conj(),E.conj()).reshape(-1,2**3,2**3)
    assert_allclose(rho, rhot)

def test_simulate_by_apply_mpo_as_CPTP():
    def return_mpo_from_2qubit_op(M):
        U, s, Vh = np.linalg.svd(M.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,-1))
        bdim = len(s[s>1e-15])
        U = np.dot(U[:,:bdim], np.diag(s[:bdim]))
        Vh = Vh[:bdim,:]
        return MPO([U.reshape(2,2,1,bdim), Vh.reshape(bdim,2,2,1).transpose(1,2,0,3)])
    
    zero = np.array([[1,0],[0,0]]).reshape(2,2,1,1)
    mpo = MPO([zero, zero])
    U = unitary_group.rvs(4)
    mpo2 = return_mpo_from_2qubit_op(U)

    mpo.apply_MPO_as_CPTP([0,1], mpo2)
    rho = mpo.contract().reshape(2**2,-1)

    state = np.array([1,0,0,0])
    state = np.dot(U, state)
    rho2 = oe.contract("i,j->ij",state,state.conj())

    assert_allclose(rho, rho2)

def test_amplitude():
    zero = np.array([1, 0])
    one = np.array([0, 1])
    A = np.random.randn(2,2,1,10)
    B = np.random.randn(2,2,10,20)
    C = np.random.randn(2,2,20,1)
    mpo = MPO([A, B, C])
    rho = mpo.contract().reshape(2,2,2,2,2,2)

    amplitude_list = [[0,1,1], [1,0,0]]
    for amp in amplitude_list:
        tensors = []
        rhotmp = rho
        for m in amp:
            if m:
                tensors.append(one)
                rhotmp = rhotmp[1]
            else:
                tensors.append(zero)
                rhotmp = rhotmp[0]
        for m in amp:
            if m:
                tensors.append(one)
                rhotmp = rhotmp[1]
            else:
                tensors.append(zero)
                rhotmp = rhotmp[0]
        amplitude = mpo.amplitude(tensors).item()

        print(amplitude, rhotmp)
        true_amplitude = rhotmp.flatten().item()

        assert_allclose(amplitude, true_amplitude)