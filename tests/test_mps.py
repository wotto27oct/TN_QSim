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
    orig_psi = mps.contract().flatten()

    # canonicalization
    mps.canonicalization()
    virtual_dims = mps.virtual_dims
    psi = mps.contract().flatten()
    assert np.all(virtual_dims <= orig_virtual_dims)
    assert_allclose(orig_psi, psi)

def test_apply_single_mpo():
    A = np.random.randn(2,1,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,1)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,3,1)
    mpo = MPO([D])

    mps.apply_MPO([0], mpo)
    psi = mps.contract().flatten()
    psit = oe.contract("abcd,blm,fmo,ioq->lcqafi",D,A,B,C).flatten()
    assert_allclose(psi, psit)

    A = np.random.randn(2,1,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,1)
    mps = MPS([A, B, C])

    E = np.random.randn(2,2,1,1)
    mpo = MPO([E])

    mps.apply_MPO([1], mpo)
    psi = mps.contract().flatten()
    psit = oe.contract("efdg,blm,fmo,ioq->lqbei",E,A,B,C).flatten()
    assert_allclose(psi, psit)

    A = np.random.randn(2,1,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,1)
    mps = MPS([A, B, C])

    F = np.random.randn(2,2,1,4)
    mpo = MPO([F])

    mps.apply_MPO([2], mpo)
    psi = mps.contract().flatten()
    psit = oe.contract("higj,blm,fmo,ioq->lqjbfh",F,A,B,C).flatten()
    assert_allclose(psi, psit)


def test_apply_mpo():
    A = np.random.randn(2,1,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,1)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5)
    F = np.random.randn(2,2,5,1)
    mpo = MPO([D, E, F])

    mps.apply_MPO([0, 1, 2], mpo)
    psi = mps.contract().flatten()
    psit = oe.contract("abcd,efdg,higj,blm,fmo,ioq->lqaeh",D,E,F,A,B,C).flatten()
    assert_allclose(psi, psit)

    A = np.random.randn(2,2,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,2)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,3,6)
    E = np.random.randn(2,2,6,5)
    F = np.random.randn(2,2,5,3)
    mpo = MPO([D, E, F])

    mps.apply_MPO([0, 1, 2], mpo, last_dir=2)
    psi = mps.contract().reshape(6, 2**3, 6)
    psit = oe.contract("abcd,efdg,higj,blm,fmo,ioq->lcqjaeh",D,E,F,A,B,C).reshape(6, 2**3, 6)
    assert_allclose(psi, psit)


def test_apply_mpo_inverse_direction():
    A = np.random.randn(2,1,3)
    B = np.random.randn(2,3,4)
    C = np.random.randn(2,4,1)
    mps = MPS([A, B, C])

    D = np.random.randn(2,2,1,6)
    E = np.random.randn(2,2,6,5)
    mpo = MPO([D, E])

    mps.apply_MPO([1,0], mpo, last_dir=2)
    psi = mps.contract().reshape(-1, 2**3)
    psit = oe.contract("efgd,abdc,blm,fmo,ioq->lcqaei",D,E,A,B,C).reshape(-1,2**3)
    assert_allclose(psi, psit)

"""
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

        assert_allclose(amplitude, true_amplitude)"""