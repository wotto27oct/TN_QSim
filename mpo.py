from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
from general_tn import TensorNetwork

class MPO(TensorNetwork):
    """class of MPO

    physical bond: (up) 0, 1, ..., n-1, (down) n, ..., 2n-1
    virtual bond: 2n, 2n+1, ..., 3n

    Attributes:
        n (int) : the number of tensors
        apex (int) : apex point of canonical form
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
        truncate_dim (int) : truncation dim of virtual bond, default None
        threthold_err (float) : the err threthold of singular values we keep
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


    def canonicalization(self):
        """canonicalize MPS
        apex point = self.0

        """
        self.apex = 0
        for i in range(self.n-1):
            self.__move_right_canonical()
        for i in range(self.n-1):
            self.__move_left_canonical()

    
    def contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        output_edge_order = [cp_nodes[0][2], cp_nodes[self.n-1][3]]
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][1])
        return tn.contractors.auto(cp_nodes, output_edge_order=output_edge_order).tensor


    def apply_gate(self, tidx, gtensor):
        """ apply nqubit gate
        
        Args:
            tidx (list of int) : list of qubit index we apply to. the apex is to be the last index.
            gtensor (np.array) : gate tensor, shape must be (pdim, pdim, ..., pdim, pdim, ...)

        Return:
            fidelity (float) : approximation accuracy as fidelity
        """

        # apexをtidx[0]に合わせる
        if self.apex is not None:
            if tidx[0] < self.apex:
                for _ in range(self.apex - tidx[0]):
                    self.__move_left_canonical()
            elif tidx[0] > self.apex:
                for _ in range(tidx[0] - self.apex):
                    self.__move_right_canonical()
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1] - tidx[0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1] - tidx[i] != 1 or not is_direction_right and tidx[i+1] - tidx[i] != -1:
                raise ValueError("gate must be applied in sequential to MPS")
        
        reshape_list = []
        for i in tidx:
            reshape_list.append(self.nodes[i][0].dimension)
        reshape_list = reshape_list + reshape_list
        gate = tn.Node(gtensor.reshape(reshape_list))
        gate_conj = tn.Node(gtensor.conj().reshape(reshape_list))
        for i in range(len(tidx)):
            self.nodes[tidx[i]][0] ^ gate[i+len(tidx)]
            self.nodes[tidx[i]][1] ^ gate_conj[i+len(tidx)]

        node_edges = []
        for i in range(len(tidx)):
            node_edges.append(gate[i])
            gate[i].set_name(f"edge {tidx[i]}")
        for i in range(len(tidx)):
            node_edges.append(gate_conj[i])
            gate_conj[i].set_name(f"edge {tidx[i]+self.n}")
        if is_direction_right:
            node_edges.append(self.nodes[tidx[0]][2])
            node_edges.append(self.nodes[tidx[-1]][3])
        else:
            node_edges.append(self.nodes[tidx[0]][3])
            node_edges.append(self.nodes[tidx[-1]][2])

        tmp = tn.contractors.optimal([self.nodes[i] for i in tidx] + [gate] + [gate_conj], ignore_edge_order=True)
        inner_edge = node_edges[-2]

        total_fidelity = 1.0

        for i in range(len(tidx)-1):
            left_edges = []
            right_edges = []
            left_edges.append(node_edges[i])
            left_edges.append(node_edges[i+len(tidx)])
            left_edges.append(inner_edge)
            for j in range(len(tidx)-1-i):
                right_edges.append(node_edges[i+j+1])
                right_edges.append(node_edges[i+j+1+len(tidx)])
            right_edges.append(node_edges[-1])
            U, s, Vh, trun_s = tn.split_node_full_svd(tmp, left_edges, right_edges, self.truncate_dim, self.threthold_err)
            U_reshape_edges = [node_edges[i], node_edges[i+len(tidx)], inner_edge, s[0]] if is_direction_right else [node_edges[i], node_edges[i+len(tidx)], s[0], inner_edge]
            self.nodes[tidx[i]] = U.reorder_edges(U_reshape_edges)
            inner_edge = s[0]
            tmp = tn.contractors.optimal([s, Vh], ignore_edge_order=True)

            self.nodes[tidx[i]].set_name(f"node {tidx[i]}")
            if is_direction_right:
                self.nodes[tidx[i]][2].set_name(f"edge {tidx[i]+2*self.n+1}")
            else:
                self.nodes[tidx[i]][1].set_name(f"edge {tidx[i]+2*self.n}")
            
            fidelity = 1.0 - np.dot(trun_s, trun_s)
            total_fidelity *= fidelity

        
        U_reshape_edges = [node_edges[len(tidx)-1], node_edges[2*len(tidx)-1], inner_edge, node_edges[-1]] if is_direction_right else [node_edges[len(tidx)-1], node_edges[2*len(tidx)-1], node_edges[-1], inner_edge]
        self.nodes[tidx[-1]] = tmp.reorder_edges(U_reshape_edges)
        self.nodes[tidx[-1]].set_name(f"node {tidx[-1]}")

        if self.apex is not None:
            self.apex = tidx[-1]
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(total_fidelity)
        
        return total_fidelity


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

    def apply_single_qubit_CPTP(self, tidx, gtensor):
        """ apply single qubit CPTP
        
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
        
        self.tensors[tidx] = oe.contract("abcd,efab->efcd", self.tensors[tidx], gtensor)
        self.edge_dims[tidx] = self.tensors[tidx].shape[0]
        self.edge_dims[tidx+self.n] = self.tensors[tidx].shape[1]

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
            # in the case of [i, i+1]
            whole_tensor = oe.contract("acef,bdfg,hiab,jkcd->hjeikg", self.tensors[left_idx], self.tensors[left_idx+1], gtensor, gtensor.conj())
        else:
            whole_tensor = oe.contract("acef,bdfg,ihba,kjdc->hjeikg", self.tensors[left_idx], self.tensors[left_idx+1], gtensor, gtensor.conj())
        reshape_dim = whole_tensor.shape[0] * whole_tensor.shape[1] * whole_tensor.shape[2]
        U, s, Vh = np.linalg.svd(whole_tensor.reshape(reshape_dim, -1), full_matrices=False)
        virtual_dim = s.shape[0]
        if self.truncate_dim is not None:
            virtual_dim = self.truncate_dim
        else:
            virtual_dim = min((i for i in range(s.shape[0]) if s[i] < self.threthold), default=s.shape[0])
            #print(s.shape[0], virtual_dim)
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

    def apply_2qubit_CPTP(self, tidx, gtensor, is_finishing_right=True):
        """ apply 2qubit CPTP
        
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

    def sample(self, seed=0):
        """ sample from mpo
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
        left_tensor = oe.contract("aacd->cd", self.tensors[0])
        for i in range(1, self.n):
            left_tensor = oe.contract("ec,aacd->ed", left_tensor, self.tensors[i])
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
