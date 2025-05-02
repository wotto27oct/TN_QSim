import opt_einsum as oe
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import unitary_group

from tn_qsim.mps import MPS
from tn_qsim.mpo import MPO

def test_contract():
    A = np.random.randn(2,1,10)
    B = np.random.randn(2,10,20)
    C = np.random.randn(2,20,1)
    mps = MPS([A, B, C])

    # contraction
    psi = mps.contract().flatten() 
    psit = oe.contract("eab,fbc,gcd->efg",A,B,C).flatten()
    assert_allclose(psi, psit)

    # replace
    D = np.random.randn(2,1,20)
    E = np.random.randn(2,20,20)
    mps.replace_tensors([0, 1], [D, E])
    psi = mps.contract().flatten() 
    psit = oe.contract("eab,fbc,gcd->efg",D,E,C).flatten()
    assert_allclose(psi, psit)

def test_canonicalization():
    A = np.random.randn(2,1,10)
    B = np.random.randn(2,10,20)
    C = np.random.randn(2,20,1)
    mps = MPS([A, B, C])
    orig_virtual_dims = mps.virtual_dims
    orig_psi= mps.contract().flatten()

    # canonicalization
    mps.canonicalization()
    virtual_dims = mps.virtual_dims
    psi = mps.contract().flatten()
    assert np.all(virtual_dims <= orig_virtual_dims)
    assert_allclose(orig_psi, psi)

def test_apply_single_mpo():
    A = np.random.randn(2,3,3)
    mps = MPS([A])

    B = np.random.randn(2,2,4,5)
    mpo = MPO([B])

    mps.apply_MPO([0], mpo)
    psi = mps.contract()
    psit = oe.contract("abc,AaBC->bBcCA",A,B).reshape(12,15,2)
    assert_allclose(psi, psit)

def test_apply_mpo_without_truncation():
    A = np.random.randn(2,2,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,3)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,4,6)
    E = np.random.randn(2,2,6,5)
    F = np.random.randn(2,2,5,5)
    mpo = MPO([D, E, F])

    mps.apply_MPO([0, 1, 2], mpo)
    psi = mps.contract()
    psit = oe.contract("abcd,efdg,higj,blm,fmo,ioq->lcqjaeh",D,E,F,A,B,C).reshape(8,15,2,2,2)
    assert_allclose(psi, psit)

    A = np.random.randn(2,2,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,3)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,4,6)
    E = np.random.randn(2,2,6,1)
    mpo = MPO([D, E])

    mps.apply_MPO([0, 1], mpo)
    psi = mps.contract()
    psit = oe.contract("abcd,efdg,blm,fmo,ioq->lcqaei",D,E,A,B,C).reshape(2*4,3,2,2,2)
    assert_allclose(psi, psit)

def test_apply_mpo_inverse_direction_without_truncation():
    A = np.random.randn(2,2,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,3)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,4,6)
    E = np.random.randn(2,2,6,5)
    F = np.random.randn(2,2,5,5)
    mpo = MPO([D, E, F])

    mps.apply_MPO([2,1,0], mpo)
    psi = mps.contract()
    psit = oe.contract("hijg,efgd,abdc,blm,fmo,ioq->lcqjaeh",D,E,F,A,B,C).reshape(2*5,3*4,2,2,2)
    assert_allclose(psi, psit)

    A = np.random.randn(2,2,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,3)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5)
    mpo = MPO([D, E])

    mps.apply_MPO([1,0], mpo)
    psi = mps.contract()
    psit = oe.contract("efgd,abdc,blm,fmo,ioq->lcqaei",D,E,A,B,C).reshape(2*5,3,2,2,2)
    assert_allclose(psi, psit)

def test_apply_mpo_with_truncation():
    A = np.random.randn(10,1,3)
    B = np.random.randn(10,3,4)
    C = np.random.randn(10,4,1)
    mps = MPS([A, B, C], truncate_dim=2)

    D = np.random.randn(10,10,1,6)
    E = np.random.randn(10,10,6,5)
    F = np.random.randn(10,10,5,1)
    mpo = MPO([D, E, F])

    mps.apply_MPO([0,1,2], mpo, is_truncate=True)
    virtual_dims = mps.virtual_dims
    assert np.all(virtual_dims <= np.array([2,2,2,2]))

    mps = MPS([A, B, C], threshold_err=0.2)
    mpo = MPO([D, E, F])

    mps.apply_MPO([0, 1, 2], mpo, is_truncate=True, is_normalize=True)
    psi = mps.contract().flatten()
    assert_allclose(1.0, np.linalg.norm(psi))
    psit = oe.contract("abcd,efdg,higj,blm,fmo,ioq->lcqjaeh",D,E,F,A,B,C).flatten()
    psit /= np.linalg.norm(psit)
    inner = np.dot(psi.conj(), psit)
    fid = inner.conj() * inner
    assert fid > 0.8 ** 4
    truncate_virtual_dims = mps.virtual_dims
    assert np.all(truncate_virtual_dims <= np.array([1,10,10,1]))

def test_calc_trace():
    A = np.random.randn(2,1,3) + 1j*np.random.randn(2,1,3)
    B = np.random.randn(2,3,4) + 1j*np.random.randn(2,3,4)
    C = np.random.randn(2,4,1) + 1j*np.random.randn(2,4,1)
    mps = MPS([A, B, C])
    trace = mps.calc_trace().flatten()
    psi = mps.contract().flatten()
    tracet = np.dot(psi.conj(), psi)
    assert_allclose(trace, tracet)


def test_simulate_by_apply_mpo():
    def return_mpo_from_2qubit_op(M):
        U, s, Vh = np.linalg.svd(M.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,-1))
        bdim = len(s[s>1e-15])
        U = np.dot(U[:,:bdim], np.diag(s[:bdim]))
        Vh = Vh[:bdim,:]
        return MPO([U.reshape(2,2,1,bdim), Vh.reshape(bdim,2,2,1).transpose(1,2,0,3)])
    
    zero = np.array([1,0]).reshape(2,1,1)
    mps = MPS([zero, zero])
    U = unitary_group.rvs(4)
    mpo = return_mpo_from_2qubit_op(U)

    mps.apply_MPO([0,1], mpo)
    psi = mps.contract().flatten()
    psit = np.dot(U, np.array([1,0,0,0]))

    assert_allclose(psi, psit)

def test_amplitude():
    zero = np.array([1, 0])
    one = np.array([0, 1])
    A = np.random.randn(2,1,10)
    B = np.random.randn(2,10,20)
    C = np.random.randn(2,20,1)
    mps = MPS([A, B, C])
    psi = mps.contract().reshape(2,2,2)
    
    amplitude_list = [[0,1,1], [1,0,0]]
    for amp in amplitude_list:
        tensors = []
        psitmp = psi
        for m in amp:
            if m:
                tensors.append(one)
                psitmp = psitmp[1]
            else:
                tensors.append(zero)
                psitmp = psitmp[0]

        amplitude = mps.amplitude(tensors).item()

        print(amplitude, psitmp)
        true_amplitude = psitmp

        assert_allclose(amplitude, true_amplitude)