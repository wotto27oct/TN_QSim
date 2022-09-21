import opt_einsum as oe
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import unitary_group

from tn_qsim.mera3D import MERA3D

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(linewidth=500)

def test_renormalize():
    # ternary MERA
    U = unitary_group.rvs(2**2).reshape(2,2,2,2)
    V = unitary_group.rvs(2**3).reshape(2,2,2,8)[:,:,:,:8]
    sigma = np.array([[[1j*0, 1 + 1j*0], [1 + 1j*0, 0*1j]],
        [[0*1j, -1j], [1j, 0*1j]],
        [[1 + 0*1j, 0*1j], [0*1j, -1 + 0*1j]]], dtype=np.complex128)
    zz_term = np.einsum('ij,kl->ikjl', sigma[2], sigma[2])
    x_term = np.einsum('ij,kl->ikjl', sigma[0], np.eye(2, dtype=np.complex128))
    H = -zz_term - 1.0 * x_term

    hamiltonianz = oe.contract("aA,bB->abAB", sigma[2], np.eye(2, dtype=np.complex128)).astype(np.complex128).reshape(4,4)
    assert np.linalg.norm(hamiltonianz - hamiltonianz.T.conj()) < 1e-7
    H = hamiltonianz
    H = H.reshape(2,2,2,2)

    U = U.astype(np.complex128)
    V = V.astype(np.complex128)
    H = H.astype(np.complex128)

    # support [1, 2]
    mera3D = MERA3D(6, H, [1, 2])
    mera3D.apply_isometry([2,3], [2,3], U)
    mera3D.apply_isometry([0,1,2], [1], V)
    mera3D.apply_isometry([3,4,5], [4], V)

    h_renorm = mera3D.renormalize()
    tensors = [H, U, V, V, U.conj(), V.conj(), V.conj()]
    node_list, output_edge_order = mera3D.prepare_renormalize()
    for idx in range(len(tensors)):
        assert_allclose(node_list[idx].tensor, tensors[idx])
    tn, tree = mera3D.find_renormalize_tree()
    #mera3D.visualize_renormalization(tn, tree)
    h_renorm_true = oe.contract("abcd,befg,hafj,gklm,deop,hcoq,pklr->jmqr",*tensors)
    print(h_renorm.dtype, h_renorm_true.dtype)
    assert np.linalg.norm(h_renorm - h_renorm_true) < 1e-8

    # support [2, 3]
    mera3D = MERA3D(6, H, [2, 3])
    mera3D.apply_isometry([2,3], [2,3], U)
    mera3D.apply_isometry([0,1,2], [1], V)
    mera3D.apply_isometry([3,4,5], [4], V)

    h_renorm = mera3D.renormalize()
    tensors = [H, U, V, V, U.conj(), V.conj(), V.conj()]
    node_list, output_edge_order = mera3D.prepare_renormalize()
    for idx in range(len(tensors)):
        assert_allclose(node_list[idx].tensor, tensors[idx])
    tn, tree = mera3D.find_renormalize_tree()
    #mera3D.visualize_renormalization(tn, tree)
    h_renorm_true = oe.contract("abcd,abfg,hefj,gklm,cdop,heoq,pklr->jmqr",*tensors)
    np.linalg.norm(h_renorm - h_renorm_true) < 1e-8

    # support [3, 4]
    mera3D = MERA3D(6, H, [3, 4])
    mera3D.apply_isometry([2,3], [2,3], U)
    mera3D.apply_isometry([0,1,2], [1], V)
    mera3D.apply_isometry([3,4,5], [4], V)

    h_renorm = mera3D.renormalize()
    tensors = [H, U, V, V, U.conj(), V.conj(), V.conj()]
    node_list, output_edge_order = mera3D.prepare_renormalize()
    for idx in range(len(tensors)):
        assert_allclose(node_list[idx].tensor, tensors[idx])
    tn, tree = mera3D.find_renormalize_tree()
    #mera3D.visualize_renormalization(tn, tree)
    h_renorm_true = oe.contract("abcd,eafg,hkfj,gblm,ecop,hkoq,pdlr->jmqr",*tensors)
    np.linalg.norm(h_renorm - h_renorm_true) < 1e-8