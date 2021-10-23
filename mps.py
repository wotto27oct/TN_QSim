from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
import opt_einsum as oe
import cotengra as ctg
from tensornetwork import TensorNetwork

class MPS(TensorNetwork):
    """class of MPS

    physical bond: 0, 1, ..., n-1
    virtual bond: n, n+1, ..., 2n

    Attributes:
        n (int) : the number of tensors
        apex (int) : apex point of canonical form
        edges (list of list of int) : the orderd indexes of each edge connected to each tensor
        edge_dims (dict of int) : dims of each edges
        tensors (list of np.array) : each tensor, [physical, virtual_left, virtual_right]
        truncate_dim (int) : truncation dim of virtual bond, default None
    """

    def __init__(self, tensors, truncate_dim=None):
        self.n = len(tensors)
        self.apex = None
        self.edges = []
        self.edge_dims = dict()
        self.tensors = tensors
        self.truncate_dim = truncate_dim
        for i in range(self.n):
            self.edges.append([i, i+self.n, i+self.n+1])
            self.edge_dims[i] = self.tensors[i].shape[0]
            if i != 0 and self.edge_dims[i+self.n] != self.tensors[i].shape[1]:
                    raise ValueError("the dim of virtual bond do not correspond")
            if i == 0:
                self.edge_dims[self.n] = self.tensors[i].shape[1]
            self.edge_dims[i+self.n+1] = self.tensors[i].shape[2]

    def canonicalization(self):
        """canonicalize MPS
        apex point = self.0

        """
        self.apex = 0
        for i in range(self.n-1):
            self.__move_right_canonical()
        for i in range(self.n-1):
            self.__move_left_canonical()

    def apply_single_qubit_gate(self, tidx, gtensor):
        """ apply single qubit gate
        
        Args:
            tidx (int) : qubit index we apply to
            gtensor (np.array) : gate tensor
        """
        if tidx < self.apex:
            for _ in range(self.apex - tidx):
                self.__move_left_canonical()
        elif tidx > self.apex:
            for _ in range(tidx - self.apex):
                self.__move_right_canonical()
        
        self.tensors[tidx] = oe.contract("abc,da->dbc", self.tensors[tidx], gtensor)
        self.edge_dims[tidx] = self.tensors[tidx].shape[0]

    def apply_2qubit_gate(self, tidx, gtensor, is_finishing_right=True):
        """ apply 2qubit gate
        
        Args:
            tidx (list of int) : list of qubit index we apply to
            gtensor (np.array) : gate tensor, shape must be (pdim, pdim, pdim, pdim)
            is_finishing_right (bool) : if True, set apex to the right-hand

        Return:
            fidelity (float) : approximation accuracy as fidelity
        """

        if np.abs(tidx[1] - tidx[0]) != 1:
            raise ValueError("2qubit gate must be applied to adjacent qubit")
        
        if tidx[0] < self.apex and tidx[1] < self.apex:
            for _ in range(self.apex - max(tidx[0], tidx[1])):
                self.__move_left_canonical()
        elif tidx[0] > self.apex and tidx[1] > self.apex:
            for _ in range(min(tidx[0], tidx[1]) - self.apex):
                self.__move_right_canonical()

        whole_tensor = None
        left_idx = min(tidx)
        if tidx[1] - tidx[0] == 1:
            whole_tensor = oe.contract("acd,bde,fgab->fcge", self.tensors[left_idx], self.tensors[left_idx+1], gtensor)
        else:
            whole_tensor = oe.contract("acd,bde,gfba->fcge", self.tensors[left_idx], self.tensors[left_idx+1], gtensor)
        reshape_dim = whole_tensor.shape[0] * whole_tensor.shape[1]
        U, s, Vh = np.linalg.svd(whole_tensor.reshape(reshape_dim, -1), full_matrices=False)
        virtual_dim = s.shape[0]
        if self.truncate_dim is not None:
            virtual_dim = self.truncate_dim
        if is_finishing_right:
            self.tensors[left_idx] = U[:,:virtual_dim].reshape(self.tensors[left_idx].shape[0], self.tensors[left_idx].shape[1], -1)
            self.tensors[left_idx+1] = oe.contract("ab,bc->ac", np.diag(s[:virtual_dim]), Vh[:virtual_dim]).reshape(-1, self.tensors[left_idx+1].shape[0], self.tensors[left_idx+1].shape[2]).transpose(1,0,2)
            self.apex = left_idx + 1
        else:
            self.tensors[left_idx+1] = Vh[:virtual_dim].reshape(-1, self.tensors[left_idx+1].shape[0], self.tensors[left_idx+1].shape[2]).transpose(1,0,2)
            self.tensors[left_idx] = oe.contract("ab,bc->ac", U[:,:virtual_dim], np.diag(s[:virtual_dim])).reshape(self.tensors[left_idx].shape[0], self.tensors[left_idx].shape[1], -1)
            self.apex = left_idx
        self.edge_dims[left_idx + self.n + 1] = self.tensors[left_idx].shape[2]

        fidelity = np.dot(s[:virtual_dim], s[:virtual_dim])
        self.tensors[self.apex] = self.tensors[self.apex] / np.sqrt(fidelity)
        return fidelity

    def sample(self, seed=0):
        """ sample from mps
        """
        for _ in range(self.apex, 0, -1):
            self.__move_left_canonical()

        np.random.seed(seed)

        output = []
        left_tensor = np.array([1])
        zero = np.array([1, 0])
        one = np.array([0, 1])
        for i in range(self.n):
            prob_matrix = oe.contract("abc,b,dec,e->ad", self.tensors[i], left_tensor, self.tensors[i].conj(), left_tensor.conj())
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
                left_tensor = oe.contract("abc,a,b->c", self.tensors[i], zero, left_tensor)
            else:
                output.append(1)
                left_tensor = oe.contract("abc,a,b->c", self.tensors[i], one, left_tensor)
        
        return np.array(output)

    def __move_right_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        reshape_dim = self.tensors[self.apex].shape[2]
        U, s, Vh = np.linalg.svd(self.tensors[self.apex].reshape(-1, reshape_dim), full_matrices=False)
        self.tensors[self.apex] = U.reshape([self.tensors[self.apex].shape[0], self.tensors[self.apex].shape[1], -1])
        self.tensors[self.apex+1] = np.einsum("ab,bd,cde->cae", np.diag(s), Vh, self.tensors[self.apex+1])
        self.edge_dims[self.apex+self.n+1] = self.tensors[self.apex].shape[2]
        self.apex = self.apex + 1

    def __move_left_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        reshape_dim = self.tensors[self.apex].shape[1]
        U, s, Vh = np.linalg.svd(self.tensors[self.apex].transpose(1,0,2).reshape(reshape_dim, -1), full_matrices=False)
        self.tensors[self.apex] = Vh.reshape(-1, self.tensors[self.apex].shape[0], self.tensors[self.apex].shape[2]).transpose(1,0,2)
        self.tensors[self.apex-1] = oe.contract("abc,cd,de", self.tensors[self.apex-1], U, np.diag(s))
        self.edge_dims[self.apex+self.n] = self.tensors[self.apex].shape[1]
        self.apex = self.apex - 1