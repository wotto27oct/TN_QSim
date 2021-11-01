from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
from general_tn import TensorNetwork

class MPDO(TensorNetwork):
    """class of MPDO

    physical bond (up) 0, 1, ..., n-1
    inner bond (down) n, ..., 2n-1
    virtual bond: 2n, 2n+1, ..., 3n

    Attributes:
        n (int) : the number of tensors
        apex (int) : apex point of canonical form
        edges (list of list of int) : the orderd indexes of each edge connected to each tensor
        edge_dims (dict of int) : dims of each edges
        tensors (list of np.array) : each tensor, [physical(up), inner(down), virtual_left, virtual_right]
        truncate_dim (int) : truncation dim of virtual bond, default None
    """

    def __init__(self, tensors, truncate_dim=None, threthold_err=None):
        self.n = len(tensors)
        edge_info = []
        for i in range(self.n):
            edge_info.append([i, self.n+i, 2*self.n+i, 2*self.n+i+1])
        super().__init__(edge_info, tensors)
        self.apex = None
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err


    @property
    def virtual_dims(self):
        virtual_dims = [self.nodes[0].get_dimension(2)]
        for i in range(self.n):
            virtual_dims.append(self.nodes[i].get_dimension(3))
        return virtual_dims


    @property
    def inner_dims(self):
        inner_dims = []
        for i in range(self.n):
            inner_dims.append(self.nodes[i].get_dimension(1))
        return inner_dims


    def canonicalization(self):
        """canonicalize MPDO
        apex point = self.0

        """
        self.apex = 0
        for i in range(self.n-1):
            self.__move_right_canonical()
        for i in range(self.n-1):
            self.__move_left_canonical()


    def contract(self):
        """contract and generate density operator.

        conjugate tensor is appended.

        all edges which dim is 1 is excluded.
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)
        for i in range(self.n):
            cp_nodes.append(tn.Node(cp_nodes[i].tensor.conj()))
            tn.connect(cp_nodes[i][1], cp_nodes[i+self.n][1])
            if i != 0:
                tn.connect(cp_nodes[self.n+i-1][3], cp_nodes[self.n+i][2])
        
        output_edge_order = [cp_nodes[0][2], cp_nodes[self.n][2], cp_nodes[self.n-1][3], cp_nodes[2*self.n-1][3]]
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])
        for i in range(self.n):
            output_edge_order.append(cp_nodes[self.n+i][0])
        return tn.contractors.auto(cp_nodes, output_edge_order=output_edge_order).tensor
    

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
        
        self.tensors[tidx] = oe.contract("abcd,ea->ebcd", self.tensors[tidx], gtensor)
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
            # in case of [i, i+1]
            whole_tensor = oe.contract("abcd,efdg,hiae->hbcifg", self.tensors[left_idx], self.tensors[left_idx+1], gtensor)
        else:
            whole_tensor = oe.contract("abcd,efdg,ihea->hbcifg", self.tensors[left_idx], self.tensors[left_idx+1], gtensor)
        reshape_dim = whole_tensor.shape[0] * whole_tensor.shape[1] * whole_tensor.shape[2]
        U, s, Vh = np.linalg.svd(whole_tensor.reshape(reshape_dim, -1), full_matrices=False)
        virtual_dim = s.shape[0]
        if self.truncate_dim is not None:
            virtual_dim = self.truncate_dim
        else:
            virtual_dim = min((i for i in range(s.shape[0]) if s[i] < self.threthold), default=s.shape[0])

        if is_finishing_right:
            self.tensors[left_idx] = U[:,:virtual_dim].reshape(self.tensors[left_idx].shape[0], self.tensors[left_idx].shape[1], self.tensors[left_idx].shape[2], -1)
            self.tensors[left_idx+1] = oe.contract("ab,bc->ac", np.diag(s[:virtual_dim]), Vh[:virtual_dim]).reshape(-1, self.tensors[left_idx+1].shape[0], self.tensors[left_idx+1].shape[1], self.tensors[left_idx+1].shape[3]).transpose(1,2,0,3)
            if self.apex is not None:
                self.apex = left_idx + 1
        else:
            self.tensors[left_idx+1] = Vh[:virtual_dim].reshape(-1, self.tensors[left_idx+1].shape[0], self.tensors[left_idx+1].shape[1], self.tensors[left_idx+1].shape[3]).transpose(1,2,0,3)
            self.tensors[left_idx] = oe.contract("ab,bc->ac", U[:,:virtual_dim], np.diag(s[:virtual_dim])).reshape(self.tensors[left_idx].shape[0], self.tensors[left_idx].shape[1],self.tensors[left_idx].shape[2], -1)
            if self.apex is not None:
                self.apex = left_idx
        self.edge_dims[left_idx + 2*self.n + 1] = self.tensors[left_idx].shape[3]

        fidelity = np.dot(s[:virtual_dim], s[:virtual_dim])
        #print("fid", fidelity)
        if is_finishing_right:
            self.tensors[left_idx+1] = self.tensors[left_idx+1] / self.calc_trace().flatten()[0]
        else:
            self.tensors[left_idx] = self.tensors[left_idx] / self.calc_trace().flatten()[0]

        #print("trace", self.calc_trace().flatten())
        return fidelity


    def apply_MPO_CPTP(self, tidx, mpo):
        """apply MPO CPTP
        
        Args:
            tidx (list of int) : list of sequential qubit index we apply to,The final index must be inner-connected tensor.
            mpo (MPO) : MPO we apply

        Return:
            fidelity (float) : approximation accuracy as fidelity of trace norm.
        """ 
        if self.apex is not None:
            if tidx[0] < self.apex:
                for _ in range(self.apex - tidx[0]):
                    self.__move_left_canonical()
            elif tidx[0] > self.apex:
                for _ in range(tidx[0] - self.apex):
                    self.__move_right_canonical()

        is_mpo_direction_right = True if tidx[1] - tidx[0] == 1 else False
        for i in range(len(tidx)-1):
            if is_mpo_direction_right and tidx[i+1] - tidx[i] != 1 or not is_mpo_direction_right and  tidx[i+1] - tidx[i] != -1:
                raise ValueError("MPO CPTP must be applied in sequential")

        fidelity = 1.0
        
        if is_mpo_direction_right:
            if not self.is_dangling[self.edges[tidx[0]][2]] and mpo.tensors[0].shape[2] != 1:
                raise ValueError("the edge of MPO CPTP must be dangling")

            whole_tensor_left = oe.contract("abcd,eafg->ebfcgd", self.tensors[tidx[0]], mpo.tensors[0]).reshape(mpo.tensors[0].shape[0], self.tensors[tidx[0]].shape[1], 
                                    mpo.tensors[0].shape[2]*self.tensors[tidx[0]].shape[2], mpo.tensors[0].shape[3], self.tensors[tidx[0]].shape[3])

            self.edge_dims[tidx[0] + 2*self.n] = mpo.tensors[0].shape[2]*self.tensors[tidx[0]].shape[2]

            for i in range(1, len(tidx)):
                whole_tensor = oe.contract("abcde,ifdj,fgeh->abcigjh", whole_tensor_left, mpo.tensors[i], self.tensors[tidx[i]])
                wshape = whole_tensor.shape
                whole_tensor = whole_tensor.reshape(wshape[0]*wshape[1]*wshape[2], -1)
                U, s, Vh = np.linalg.svd(whole_tensor, full_matrices=False)
                virtual_dim = s.shape[0]
                if self.truncate_dim is not None:
                    virtual_dim = self.truncate_dim
                else:
                    virtual_dim = min((i for i in range(s.shape[0]) if s[i] < self.threthold), default=s.shape[0])
                self.tensors[tidx[i-1]] = U[:,:virtual_dim].reshape(wshape[0], wshape[1], wshape[2], -1)
                whole_tensor_left = oe.contract("ab,bc->ac", np.diag(s[:virtual_dim]), Vh[:virtual_dim]).reshape(
                    -1, wshape[3], wshape[4], wshape[5], wshape[6]).transpose(1,2,0,3,4)
                if self.apex is not None:
                    self.apex = tidx[i]
                self.edge_dims[tidx[i-1] + 2*self.n + 1] = self.tensors[tidx[i-1]].shape[3]

                fidelity *= np.dot(s[:virtual_dim], s[:virtual_dim])
            
            self.tensors[tidx[-1]] = whole_tensor_left.transpose(0,3,1,2,4).reshape(
                whole_tensor_left.shape[0], whole_tensor_left.shape[3]*whole_tensor_left.shape[1], whole_tensor_left.shape[2], whole_tensor_left.shape[4])
            self.edge_dims[tidx[-1]+self.n] = self.tensors[tidx[-1]].shape[1]

        self.tensors[tidx[-1]] = self.tensors[tidx[-1]] / self.calc_trace().flatten()[0]

        return fidelity

    def sample(self, seed=0):
        """ sample from mpdo
            not implemented yet
        """
        #for _ in range(self.apex, 0, -1):
        #    self.__move_left_canonical()

        np.random.seed(seed)

        output = []
        left_tensor = np.array([1])
        #for i in range(self.n-1):
        #    left_tensor.append(oe.contract("aacd,c->d", self.tensors[i], left_tensor[i]))
        right_tensor = [np.array([1])]
        for i in range(self.n-1, 0, -1):
            right_tensor.append(oe.contract("aacd,d->c", self.tensors[i], right_tensor[self.n-1-i]))
        right_tensor = right_tensor[::-1]
        zero = np.array([1, 0])
        one = np.array([0, 1])
        for i in range(self.n):
            prob_matrix = oe.contract("abcd,c,d->ab", self.tensors[i], left_tensor, right_tensor[i])
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
                left_tensor = oe.contract("abcd,a,b,c->d", self.tensors[i], zero, zero, left_tensor)
            else:
                output.append(1)
                left_tensor = oe.contract("abcd,a,b,c->d", self.tensors[i], one, one, left_tensor)
        
        return np.array(output)

    def calc_trace(self):
        left_tensor = oe.contract("abcd,abfg->cfdg", self.tensors[0], self.tensors[0].conj())
        for i in range(1, self.n):
            left_tensor = oe.contract("hicf,abcd,abfg->hidg", left_tensor, self.tensors[i], self.tensors[i].conj())
        return left_tensor

    def __move_right_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        l_edges = self.nodes[self.apex].get_all_edges()
        r_edges = self.nodes[self.apex+1].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [l_edges[0], l_edges[1], l_edges[2]], [l_edges[3]])
        self.nodes[self.apex] = U.reorder_edges([l_edges[0], l_edges[1], l_edges[2], s[0]])
        self.nodes[self.apex+1] = tn.contractors.optimal([s, Vh, self.nodes[self.apex+1]], output_edge_order=[r_edges[0], r_edges[1], s[0], r_edges[3]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex+1].set_name(f"node {self.apex+1}")
        self.nodes[self.apex][2].set_name(f"edge {self.apex+2*self.n+1}")

        self.apex = self.apex + 1


    def __move_left_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        l_edges = self.nodes[self.apex-1].get_all_edges()
        r_edges = self.nodes[self.apex].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [r_edges[2]], [r_edges[0], r_edges[1], r_edges[3]])
        self.nodes[self.apex] = Vh.reorder_edges([r_edges[0], r_edges[1], s[1], r_edges[3]])
        self.nodes[self.apex-1] = tn.contractors.optimal([self.nodes[self.apex-1], U, s], output_edge_order=[l_edges[0], l_edges[1], l_edges[2], s[1]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex-1].set_name(f"node {self.apex-1}")
        self.nodes[self.apex][1].set_name(f"edge {self.apex+2*self.n}")

        self.apex = self.apex - 1
