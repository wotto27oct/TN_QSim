from re import L
import numpy as np
from scipy.linalg import expm

np.set_printoptions(linewidth=500)

hbar = 1.0

def annihilation(level):
    return np.diag(np.sqrt(np.arange(1, level)), k=1)

def creation(level):
    return np.diag(np.sqrt(np.arange(1, level)), k=-1)

def number(level):
    return np.diag(np.arange(level))

def displacement(level, alpha):
    alpha = complex(alpha)
    return expm(alpha*creation(level) - alpha.conjugate()*annihilation(level))

def squeezing(level, beta):
    beta = complex(beta)
    annihi_op = annihilation(level)
    cre_op = creation(level)
    return expm((beta * np.dot(annihi_op, annihi_op) - beta.conjugate() * np.dot(cre_op, cre_op)) / 2.0)

def logical_X(level):
    return displacement(level, np.sqrt(np.pi/2.0))

def logical_Z(level):
    return displacement(level, np.sqrt(np.pi/2.0) * 1.0j)

def Q(level):
    return (annihilation(level) + creation(level)) / np.sqrt(2)

def P(level):
    return 1j * (creation(level) - annihilation(level)) / np.sqrt(2)

def X(level, x):
    return expm(-1j * x * P(level) / hbar)

def Z(level, p):
    return expm(1j * p * Q(level) / hbar)

def S(level):
    Qm = Q(level)
    return expm(1j * np.dot(Qm, Qm) / 2.0)

def H(level):
    Nm = number(level)
    return expm(1j * np.pi * Nm / 2.0)

def CNOT(level):
    Qm = Q(level)
    Pm = P(level)
    return expm(-1j * np.einsum("ij,kl->ikjl", Qm, Pm).reshape(level**2, -1))

def PiX(level):
    Qm = Q(level)
    eig, v = np.linalg.eig(Qm)
    v = v.T
    pi_x = []
    for q in range(level):
        pi_x.append(np.einsum("i,j->ij",v[q],v[q].conj()))
    pi_x = [x for _,x in sorted(zip(eig, pi_x))]
    v = [x for _,x in sorted(zip(eig, v))]
    eig = np.sort(eig)
    return eig, v, pi_x

def PiP(level):
    Pm = P(level)
    eig, v = np.linalg.eig(Pm)
    v = v.T
    pi_p = []
    for q in range(level):
        #pi_p.append(np.einsum("i,j->ij",v[:,q],v[:,q].conj()))
        pi_p.append(np.einsum("i,j->ij",v[q],v[q].conj()))
    pi_p = [x for _,x in sorted(zip(eig, pi_p))]
    v = [x for _,x in sorted(zip(eig, v))]
    eig = np.sort(eig)
    return eig, v, pi_p

def logical_zero(level, nmax, gamma, kappa):
    lzero = np.zeros(level, dtype=np.complex64)
    for n in range(-nmax, nmax+1):
        zero = np.array([0 if i != 0 else 1 for i in range(level)])
        zero = np.dot(squeezing(level, gamma), zero)
        zero = np.dot(X(level, 2*n*np.sqrt(np.pi)), zero)
        print(np.dot(X(level, 2*n*np.sqrt(np.pi)), zero)[:10])
        print(n, np.linalg.norm(np.dot(expm(-kappa**2 * number(level)), zero)))
        lzero += np.dot(expm(-kappa**2 * number(level)), zero)
    return lzero / np.linalg.norm(lzero)

def logical_one(level, nmax, gamma, kappa):
    lone = np.zeros(level, dtype=np.complex64)
    for n in range(-nmax, nmax):
        one = np.array([0 if i != 0 else 1 for i in range(level)])
        one = np.dot(squeezing(level, gamma), one)
        one = np.dot(X(level, (2*n+1)*np.sqrt(np.pi)), one)
        lone += np.dot(expm(-kappa**2 * number(level)), one)
    return lone / np.linalg.norm(lone)

def appro_SX(level, kappa):
    lX = logical_X(level)
    SX = np.dot(lX, lX)
    cre_op = annihilation(level)
    annihi_op = creation(level)
    aSX = np.dot(expm(-kappa**2 * np.dot(annihi_op, cre_op)), SX)
    aSX = np.dot(aSX, expm(kappa**2 * np.dot(annihi_op, cre_op)))
    return aSX

