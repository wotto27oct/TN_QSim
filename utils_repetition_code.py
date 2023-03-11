from re import I
import numpy as np
from tn_qsim.mpo import MPO
from scipy.linalg import expm

from get_kraus import *

from jax.config import config
config.update("jax_enable_x64", True)

np.set_printoptions(linewidth=500)

zero = np.array([1,0,0]).astype(dtype=np.complex128)
one = np.array([0,1,0]).astype(dtype=np.complex128)
two = np.array([0,0,1]).astype(dtype=np.complex128)

rzero = np.atleast_2d(zero).T @ np.atleast_2d(zero).conj()
rone = np.atleast_2d(one).T @ np.atleast_2d(one).conj()
rtwo = np.atleast_2d(two).T @ np.atleast_2d(two).conj()

I = np.eye(3)
X = np.array([[0,1,0],[1,0,0],[0,0,1]])
Z = np.array([[1,0,0],[0,-1,0],[0,0,1]])
Y = -1j*np.dot(X,Z)

i2 = 1.0 / np.sqrt(2)
Hadamard = np.array([[i2,i2,0],[i2,-i2,0],[0,0,1]]).astype(dtype=np.complex128)

def return_mpo_from_2qubit_op(M, dim1=3, dim2=3, bdim=None):
    U, s, Vh = np.linalg.svd(M.reshape(dim1,dim2,dim1,dim2).transpose(0,2,1,3).reshape(dim1**2,-1))
    if bdim is None:
        bdim = len(s[s>1e-15])
    U = np.dot(U[:,:bdim], np.diag(s[:bdim]))
    Vh = Vh[:bdim,:]
    return MPO([U.reshape(dim1,dim1,1,bdim), Vh.reshape(bdim,dim2,dim2,1).transpose(1,2,0,3)])

def return_abcd(theta, phi, lam):
    Umat = np.exp(-0.5j*theta) * expm(0.5j*theta*
        (np.cos(phi)*Z + np.sin(phi)*(np.cos(lam)*X+np.sin(lam)*Y)))
    return Umat[0,0], Umat[0,1], Umat[1,0], Umat[1,1]

def R02(theta, phi, lam):
    a, b, c, d = return_abcd(theta, phi, lam)
    return np.array([[a,0,b],[0,1,0],[c,0,d]])

def R12(theta, phi, lam):
    a, b, c, d = return_abcd(theta, phi, lam)
    return np.array([[1,0,0],[0,a,b],[0,c,d]])

def RZ(varphi):
    return np.array([[1,0,0],[0,1,0],[0,0,np.exp(1j*varphi)]])

"""def Rot(theta0, theta1, theta2, theta3, phi, lam):
    R02_0 = R02(theta0, phi, lam)
    R12_1 = R12(theta1, phi, lam)
    R02_2 = R02(theta2, phi, lam)
    R12_3 = R12(theta3, phi, lam)
    return np.kron(np.dot(R02_0, R12_1), np.dot(R02_2, R12_3))"""

def Rot(theta, phi, lam, varphi):
    R02_0 = R02(theta, phi, lam)
    R12_1 = R12(theta, phi, lam)
    Rz = RZ(varphi)
    return np.dot(Rz, np.dot(R02_0, R12_1))

def AD(p):
    K0 = np.array([[1,0,0],[0,np.sqrt(1-p),0],[0,0,1-p]])
    K1 = np.array([[0,np.sqrt(p),0],[0,0,np.sqrt(2*(1-p)*p)],[0,0,0]])
    K2 = np.array([[0,0,p],[0,0,0],[0,0,0]])
    return K0, K1, K2

def ADCP(p):
    K0, K1, K2 = AD(p)
    E = np.array([K0, K1, K2]).transpose(1,2,0).reshape(3,3,1,3)
    return E

def DP(p):
    K0 = np.sqrt(1-p) * I
    K1 = np.sqrt(p/3) * X
    K2 = np.sqrt(p/3) * Y
    K3 = np.sqrt(p/3) * Z
    return K0, K1, K2, K3

def DPCP(p):
    K0, K1, K2, K3 = DP(p)
    assert np.allclose(np.dot(K0,K0.T.conj()) + np.dot(K1,K1.T.conj()) + np.dot(K2,K2.T.conj()) + np.dot(K3,K3.T.conj()), np.eye(3))
    E = np.array([K0, K1, K2, K3]).transpose(1,2,0).reshape(3,3,1,4)
    return E

def krausCP(gamma):
    gamma = gamma # in [MHz]
    kbT_over_hw = 0.5 # kb T / hbar w [a.u.]
    tau = 1.0 # in [us]
    dim = 3
    eps = 1e-13
    kraus_list = get_kraus_list(gamma, kbT_over_hw, tau, dim, eps)
    I = np.zeros((3, 3), dtype=np.complex128)
    for K in kraus_list:
        I += np.dot(K.T.conj(), K)
    assert np.allclose(I, np.eye(3))
    E = np.array(kraus_list).transpose(1,2,0).reshape(3,3,1,len(kraus_list))
    return E

def krauslist(gamma):
    gamma = gamma # in [MHz]
    kbT_over_hw = 0.5 # kb T / hbar w [a.u.]
    tau = 1.0 # in [us]
    dim = 3
    eps = 1e-13
    kraus_list = get_kraus_list(gamma, kbT_over_hw, tau, dim, eps)
    I = np.zeros((3, 3), dtype=np.complex128)
    for K in kraus_list:
        I += np.dot(K.T.conj(), K)
    assert np.allclose(I, np.eye(3))
    return kraus_list

def krauslist_by_gamma_tau_temp(gamma, tau, temp):
    gamma = gamma # in [MHz]
    temp = temp # in [mK]
    kbT_over_hw = 13.1 * temp / 1000.0 # kb T / hbar w [a.u.]
    tau = tau # in [us]
    dim = 3
    eps = 1e-13
    kraus_list = get_kraus_list(gamma, kbT_over_hw, tau, dim, eps)
    I = np.zeros((3, 3), dtype=np.complex128)
    for K in kraus_list:
        I += np.dot(K.T.conj(), K)
    assert np.allclose(I, np.eye(3))
    return kraus_list


def measCP(pm):
    Pi0 = np.array([[1,0,0],[0,0,0],[0,0,0]])
    Pi1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    Pi2 = np.array([[0,0,0],[0,0,0],[0,0,1]])
    E0 = np.array([Pi0, pm*Pi2]).transpose(1,2,0).reshape(3,3,1,2)
    E1 = np.array([Pi1, (1-pm)*Pi2]).transpose(1,2,0).reshape(3,3,1,2)
    return E0, E1

def create_logical_cnot(qnum):
    tensor_list = []
    for l in range(qnum):
        if l == 0:
            Q = np.zeros([2,3,3])
            Q[0] = np.eye(3)
            Q[1] = X
            Q = Q.transpose(1,2,0)
            Q = Q.reshape(3,3,1,2)
        elif l % 2 == 1:
            Q = np.zeros([2,2,3,3])
            Q[0][0] = np.eye(3)
            Q[1][1] = np.eye(3)
            Q = Q.transpose(2,3,0,1)
        else:
            Q = np.zeros([2,2,3,3])
            Q[0][0] = np.eye(3)
            Q[1][1] = X
            Q = Q.transpose(2,3,0,1)
        tensor_list.append(Q)
    mpo = MPO(tensor_list)
    return mpo

def zero_if_close(value, threshold=1e-14):
    return 0 if value < threshold else value

CNOT3 = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                    [0,0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]])
CNOT_mpo = return_mpo_from_2qubit_op(CNOT3)

dqnum5_loookup_table = [None for _ in range(2**4)]
dqnum5_loookup_table[int("0b0000", 0)] = [0,0,0,0,0]
dqnum5_loookup_table[int("0b0001", 0)] = [0,0,0,0,1]
dqnum5_loookup_table[int("0b0010", 0)] = [0,0,0,1,1]
dqnum5_loookup_table[int("0b0011", 0)] = [0,0,0,1,0]
dqnum5_loookup_table[int("0b0100", 0)] = [1,1,0,0,0]
dqnum5_loookup_table[int("0b0101", 0)] = [0,0,1,1,0]
dqnum5_loookup_table[int("0b0110", 0)] = [0,0,1,0,0]
dqnum5_loookup_table[int("0b0111", 0)] = [0,0,1,0,1]
dqnum5_loookup_table[int("0b1000", 0)] = [1,0,0,0,0]
dqnum5_loookup_table[int("0b1001", 0)] = [1,0,0,0,1]
dqnum5_loookup_table[int("0b1010", 0)] = [0,1,1,0,0]
dqnum5_loookup_table[int("0b1011", 0)] = [1,0,0,1,0]
dqnum5_loookup_table[int("0b1100", 0)] = [0,1,0,0,0]
dqnum5_loookup_table[int("0b1101", 0)] = [0,1,0,0,1]
dqnum5_loookup_table[int("0b1110", 0)] = [1,0,1,0,0]
dqnum5_loookup_table[int("0b1111", 0)] = [0,1,0,1,0]