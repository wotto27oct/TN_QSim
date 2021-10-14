from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
import opt_einsum as oe
import cotengra as ctg
from tensornetwork import TensorNetwork

class MPO(TensorNetwork):
    """class of MPS

    physical bond: 0, 1, ..., n-1, n, ..., 2n-1
    virtual bond: 2n, 2n+1, ..., 3n

    Attributes:
        n (int) : the number of tensors
        apex (int) : apex point of canonical form
        edges (list of list of int) : the orderd indexes of each edge connected to each tensor
        edge_dims (dict of int) : dims of each edges
        tensors (list of np.array) : each tensor, [physical1, physical2, virtual_left, virtual_right]
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
            self.edges.append([i, i+self.n, i+2*self.n, i+2*self.n+1])
            self.edge_dims[i] = self.tensors[i].shape[0]
            self.edge_dims[i+self.n] = self.tensors[i].shape[1]
            if i != 0 and self.edge_dims[i+2*self.n] != self.tensors[i].shape[2]:
                    raise ValueError("the dim of virtual bond do not correspond")
            if i == 0:
                self.edge_dims[2*self.n] = self.tensors[i].shape[2]
            self.edge_dims[i+2*self.n+1] = self.tensors[i].shape[3]

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
        if self.apex is not None:
            if tidx < self.apex:
                for _ in range(self.apex - tidx):
                    self.__move_left_canonical()
            elif tidx > self.apex:
                for _ in range(tidx - self.apex):
                    self.__move_right_canonical()
        
        self.tensors[tidx] = oe.contract("abcd,ea,fb->efcd", self.tensors[tidx], gtensor, gtensor.conj())
        self.edge_dims[tidx] = self.tensors[tidx].shape[0]
        self.edge_dims[tidx+self.n] = self.tensors[tidx].shape[1]

    def apply_2qubit_CPTP(self, tidx, gtensor, is_finising_right=True):
        """ apply 2qubit gate
        
        Args:
            tidx (list of int) : list of qubit index we apply to
            gtensor (np.array) : gate tensor, shape must be (pdim, pdim, pdim, pdim, pdim, pdim, pdim, pdim)
            is_finishing_right (bool) : if True, set apex to the right-hand

        Return:
            fidelity (float) : approximation accuracy as fidelity
        """

        if np.abs(tidx[1] - tidx[0]) != 1:
            raise ValueError("2qubit gate must be applied to adjacent qubit")
        
        if self.apex is not None:
            if tidx[0] < self.apex and tidx[1] < self.apex:
                for _ in range(self.apex - max(tidx[0], tidx[1])):
                    self.__move_left_canonical()
            elif tidx[0] > self.apex and tidx[1] > self.apex:
                for _ in range(min(tidx[0], tidx[1]) - self.apex):
                    self.__move_right_canonical()

        whole_tensor = None
        left_idx = min(tidx)
        if tidx[1] - tidx[0] == 1:
            # in the case of [i, i+1]
            whole_tensor = oe.contract("acef,bdfg,hijkabcd->hjeikg", self.tensors[left_idx], self.tensors[left_idx+1], gtensor)
        else:
            whole_tensor = oe.contract("acef,bdfg,ihkjbadc->hjeikg", self.tensors[left_idx], self.tensors[left_idx+1], gtensor)
        reshape_dim = whole_tensor.shape[0] * whole_tensor.shape[1] * whole_tensor.shape[2]
        U, s, Vh = np.linalg.svd(whole_tensor.reshape(reshape_dim, -1), full_matrices=False)
        virtual_dim = s.shape[0]
        if self.truncate_dim is not None:
            virtual_dim = self.truncate_dim
        if is_finising_right:
            self.tensors[left_idx] = U[:,:virtual_dim].reshape(self.tensors[left_idx].shape[0], self.tensors[left_idx].shape[1], self.tensors[left_idx].shape[2], -1)
            self.tensors[left_idx+1] = oe.contract("ab,bc->ac", np.diag(s[:virtual_dim]), Vh[:virtual_dim]).reshape(-1, self.tensors[left_idx+1].shape[0], self.tensors[left_idx+1].shape[1], self.tensors[left_idx+1].shape[3]).transpose(1,2,0,3)
            if self.apex is not None:
                self.apex = left_idx + 1
        else:
            self.tensors[left_idx+1] = Vh[:virtual_dim].reshape(-1, self.tensors[left_idx+1].shape[0], self.tensors[left_idx+1].shape[1], self.tensors[left_idx+1].shape[3]).transpose(2,0,1,3)
            self.tensors[left_idx] = oe.contract("ab,bc->ac", U[:,:virtual_dim], np.diag(s[:virtual_dim])).reshape(self.tensors[left_idx].shape[0], self.tensors[left_idx].shape[1],self.tensors[left_idx].shape[2], -1)
            if self.apex is not None:
                self.apex = left_idx
        self.edge_dims[left_idx + 2*self.n + 1] = self.tensors[left_idx].shape[3]

        fidelity = np.dot(s[:virtual_dim], s[:virtual_dim])
        print("fid", fidelity)
        if is_finising_right:
            self.tensors[left_idx+1] = self.tensors[left_idx+1] / self.calc_trace().flatten()[0]
        else:
            self.tensors[left_idx] = self.tensors[left_idx] / self.calc_trace().flatten()[0]

        print("trace", self.calc_trace().flatten())
        return fidelity

    def sample(self, seed=0):
        """ sample from mpo
        """
        for _ in range(self.apex, 0, -1):
            self.__move_left_canonical()

        np.random.seed(seed)

        output = []
        left_tensor = [np.array([1])]
        for i in range(self.n-1):
            left_tensor.append(oe.contract("aacd,c->d", self.tensors[i], left_tensor[i]))
        right_tensor = [np.array([1])]
        for i in range(self.n-1, 0, -1):
            right_tensor.append(oe.contract("aacd,d->c", self.tensors[i], left_tensor[i]))
        right_tensor = right_tensor[::-1]
        zero = np.array([1, 0])
        one = np.array([0, 1])
        for i in range(self.n):
            prob_matrix = oe.contract("abcd,c,d->ab", self.tensors[i], left_tensor[i], right_tensor[i])
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
            else:
                output.append(1)
        
        return np.array(output)

    def calc_trace(self):
        left_tensor = oe.contract("aacd->cd", self.tensors[0])
        for i in range(1, self.n):
            left_tensor = oe.contract("ec,aacd->ed", left_tensor, self.tensors[i])
        return left_tensor

    def __move_right_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        reshape_dim = self.tensors[self.apex].shape[3]
        U, s, Vh = np.linalg.svd(self.tensors[self.apex].reshape(-1, reshape_dim), full_matrices=False)
        self.tensors[self.apex] = U.reshape([self.tensors[self.apex].shape[0], self.tensors[self.apex].shape[1], self.tensors[self.apex].shape[2], -1])
        self.tensors[self.apex+1] = oe.contract("ab,be,cdef->cdaf", np.diag(s), Vh, self.tensors[self.apex+1])
        self.edge_dims[self.apex+2*self.n+1] = self.tensors[self.apex].shape[3]
        self.apex = self.apex + 1

    def __move_left_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        reshape_dim = self.tensors[self.apex].shape[2]
        U, s, Vh = np.linalg.svd(self.tensors[self.apex].transpose(2,0,1,3).reshape(reshape_dim, -1), full_matrices=False)
        self.tensors[self.apex] = Vh.reshape(-1, self.tensors[self.apex].shape[0], self.tensors[self.apex].shape[1], self.tensors[self.apex].shape[3]).transpose(1,2,0,3)
        self.tensors[self.apex-1] = oe.contract("abcd,de,ef", self.tensors[self.apex-1], U, np.diag(s))
        self.edge_dims[self.apex+2*self.n] = self.tensors[self.apex].shape[2]
        self.apex = self.apex - 1