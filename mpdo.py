from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
from numpy.core.numeric import full
from scipy.linalg.special_matrices import fiedler
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

    
    def apply_gate(self, tidx, gtensor, full_update=False):
        """ apply nqubit gate
        trace-norm optimal truncation is used. full-update is not implemented yet.
        
        Args:
            tidx (list of int) : list of qubit index we apply to. the apex is to be the last index.
            gtensor (np.array) : gate tensor, shape must be (pdim, pdim, ..., pdim, pdim, ...)

        Return:
            fidelity (float) : approximation accuracy as fidelity
        """

        if full_update:
            raise ValueError("full-update for MPDO is not implemented yet.")

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
                raise ValueError("gate must be applied in sequential to MPDO")
        
        reshape_list = []
        for i in tidx:
            reshape_list.append(self.nodes[i][0].dimension)
        reshape_list = reshape_list + reshape_list
        gate = tn.Node(gtensor.reshape(reshape_list))
        for i in range(len(tidx)):
            self.nodes[tidx[i]][0] ^ gate[i+len(tidx)]

        node_edges = []
        for i in range(len(tidx)):
            node_edges.append(gate[i])
            gate[i].set_name(f"edge {tidx[i]}")
        for i in range(len(tidx)):
            node_edges.append(self.nodes[tidx[i]][1])
        if is_direction_right:
            node_edges.append(self.nodes[tidx[0]][2])
            node_edges.append(self.nodes[tidx[-1]][3])
        else:
            node_edges.append(self.nodes[tidx[0]][3])
            node_edges.append(self.nodes[tidx[-1]][2])

        tmp = tn.contractors.optimal([self.nodes[i] for i in tidx] + [gate], ignore_edge_order=True)
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
                self.nodes[tidx[i]][3].set_name(f"edge {tidx[i]+2*self.n+1}")
            else:
                self.nodes[tidx[i]][2].set_name(f"edge {tidx[i]+2*self.n}")
            
            fidelity = 1.0 - np.dot(trun_s, trun_s)
            total_fidelity *= fidelity

        
        U_reshape_edges = [node_edges[len(tidx)-1], node_edges[2*len(tidx)-1], inner_edge, node_edges[-1]] if is_direction_right else [node_edges[len(tidx)-1], node_edges[2*len(tidx)-1], node_edges[-1], inner_edge]
        self.nodes[tidx[-1]] = tmp.reorder_edges(U_reshape_edges)
        self.nodes[tidx[-1]].set_name(f"node {tidx[-1]}")

        if self.apex is not None:
            self.apex = tidx[-1]
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(total_fidelity)
        else:
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(self.calc_trace().flatten()[0])

        return total_fidelity

    
    def apply_CPTP(self, tidx, gtensor, full_update=False, inner_full_update=False):
        """ apply nqubit gate
        trace-norm optimal truncation is used. full-update is not implemented yet.
        
        Args:
            tidx (list of int) : list of qubit index we apply to. the apex is to be the last index.
            gtensor (np.array) : gate tensor, receive (Aac) tensor applied to (a) state. shape must be (new_pdim, new_pdim, ..., old_pdim, old_pdim, ..., inner_dim).

        Return:
            fidelity (float) : approximation accuracy as fidelity
        """

        if full_update:
            raise ValueError("full-update for MPDO is not implemented yet.")
        if inner_full_update:
            raise ValueError("inner-full-update for MPDO is not implemented yet.")

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
                raise ValueError("gate must be applied in sequential to MPDO")
        
        gate = tn.Node(gtensor)
        for i in range(len(tidx)):
            self.nodes[tidx[i]][0] ^ gate[i+len(tidx)]

        node_edges = []
        for i in range(len(tidx)):
            node_edges.append(gate[i])
            gate[i].set_name(f"edge {tidx[i]}")
        for i in range(len(tidx)):
            node_edges.append(self.nodes[tidx[i]][1])
        if is_direction_right:
            node_edges.append(self.nodes[tidx[0]][2])
            node_edges.append(self.nodes[tidx[-1]][3])
        else:
            node_edges.append(self.nodes[tidx[0]][3])
            node_edges.append(self.nodes[tidx[-1]][2])

        merge_edge_list = [self.nodes[tidx[-1]][1], gate[2*len(tidx)]]

        tmp = tn.contractors.optimal([self.nodes[i] for i in tidx] + [gate], ignore_edge_order=True)
        merged_edge = tn.flatten_edges(merge_edge_list)
        merged_edge.set_name(f"edge {tidx[-1]+self.n}")
        node_edges[2*len(tidx)-1] = merged_edge
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
                self.nodes[tidx[i]][3].set_name(f"edge {tidx[i]+2*self.n+1}")
            else:
                self.nodes[tidx[i]][2].set_name(f"edge {tidx[i]+2*self.n}")
            
            fidelity = 1.0 - np.dot(trun_s, trun_s)
            total_fidelity *= fidelity
        
        U_reshape_edges = [node_edges[len(tidx)-1], node_edges[2*len(tidx)-1], inner_edge, node_edges[-1]] if is_direction_right else [node_edges[len(tidx)-1], node_edges[2*len(tidx)-1], node_edges[-1], inner_edge]
        self.nodes[tidx[-1]] = tmp.reorder_edges(U_reshape_edges)
        self.nodes[tidx[-1]].set_name(f"node {tidx[-1]}")

        # heuristic simple-update for inner dimension
        shape_list = self.nodes[tidx[-1]].tensor.shape
        if shape_list[0] * shape_list[2] * shape_list[3] < shape_list[1]:
            tmp = oe.contract("abcd,ebfg->acdefg", self.nodes[tidx[-1]].tensor, self.nodes[tidx[-1]].tensor.conj())
            U, s, Vh = np.linalg.svd(tmp.reshape(shape_list[0]*shape_list[2]*shape_list[3], -1), full_matrices=False)
            self.nodes[tidx[-1]].set_tensor(oe.contract("ab,bc->ac", U, np.diag(np.sqrt(s))).reshape(shape_list[0], -1, shape_list[2], shape_list[3]))

        if self.apex is not None:
            self.apex = tidx[-1]
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(total_fidelity)
        else:
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(self.calc_trace().flatten()[0])

        return total_fidelity


    def sample(self, seed=0):
        """ sample from mpdo
            not implemented yet
        """
        #for _ in range(self.apex, 0, -1):
        #    self.__move_left_canonical()

        np.random.seed(seed)

        output = []
        left_tensor = np.array([1]).reshape(1,1)
        #for i in range(self.n-1):
        #    left_tensor.append(oe.contract("aacd,c->d", self.tensors[i], left_tensor[i]))
        right_tensor = [np.array([1]).reshape(1,1)]
        for i in range(self.n-1, 0, -1):
            right_tensor.append(oe.contract("abcd,abfg,dg->cf", self.nodes[i].tensor, self.nodes[i].tensor.conj(), right_tensor[self.n-1-i]))
        right_tensor = right_tensor[::-1]
        zero = np.array([1, 0])
        one = np.array([0, 1])
        for i in range(self.n):
            prob_matrix = oe.contract("abcd,ebfg,cf,dg->ae", self.nodes[i].tensor, self.nodes[i].tensor.conj(), left_tensor, right_tensor[i])
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
                left_tensor = oe.contract("abcd,ebfg,cf,a,e->dg", self.nodes[i].tensor, self.nodes[i].tensor.conj(), left_tensor, zero.conj(), zero)
            else:
                output.append(1)
                left_tensor = oe.contract("abcd,ebfg,cf,a,e->dg", self.nodes[i].tensor, self.nodes[i].tensor.conj(), left_tensor, one.conj(), one)
        
        return np.array(output)

    def calc_trace(self):
        left_tensor = oe.contract("abcd,abfg->cfdg", self.nodes[0].tensor, self.nodes[0].tensor.conj())
        for i in range(1, self.n):
            left_tensor = oe.contract("hicf,abcd,abfg->hidg", left_tensor, self.nodes[i].tensor, self.nodes[i].tensor.conj())
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
        self.nodes[self.apex][3].set_name(f"edge {self.apex+2*self.n+1}")

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
        self.nodes[self.apex][2].set_name(f"edge {self.apex+2*self.n}")

        self.apex = self.apex - 1