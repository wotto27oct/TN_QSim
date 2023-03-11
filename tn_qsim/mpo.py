import numpy as np
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork

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
        threshold_err (float) : the err threshold of singular values we keep
    """

    def __init__(self, tensors, truncate_dim=None, threshold_err=None):
        self.n = len(tensors)
        edge_info = []
        for i in range(self.n):
            edge_info.append([i, self.n+i, 2*self.n+i, 2*self.n+i+1])
        super().__init__(edge_info, tensors)
        self.apex = None
        self.truncate_dim = truncate_dim
        self.threshold_err = threshold_err

    @property
    def virtual_dims(self):
        virtual_dims = [self.nodes[0].get_dimension(2)]
        for i in range(self.n):
            virtual_dims.append(self.nodes[i].get_dimension(3))
        return virtual_dims

    def canonicalization(self, threshold=1.0):
        """canonicalize MPO
        apex point is set to be self.0

        Args:
            threshold (float) : truncation threshold for svd
        """
        self.apex = 0
        for i in range(self.n-1):
            self.move_right_canonical(threshold=1.0)
        for i in range(self.n-1):
            self.move_left_canonical(threshold)

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
                    self.move_left_canonical()
            elif tidx[0] > self.apex:
                for _ in range(tidx[0] - self.apex):
                    self.move_right_canonical()
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1] - tidx[0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1] - tidx[i] != 1 or not is_direction_right and tidx[i+1] - tidx[i] != -1:
                raise ValueError("gate must be applied in sequential to MPO")
        
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
            U, s, Vh, trun_s = tn.split_node_full_svd(tmp, left_edges, right_edges, self.truncate_dim, self.threshold_err)
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
        
        self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / self.calc_trace().flatten()[0]
        
        return total_fidelity

    def apply_CPTP(self, tidx, gtensor):
        """ apply nqubit gate
        
        Args:
            tidx (list of int) : list of qubit index we apply to. the apex is to be the last index.
            gtensor (np.array) : gate tensor, receive (ABab) tensor applied to (ab) state. shape must be (new_pdim, new_pdim, ..., new_pdim, new_pdim, ..., old_pdim, old_pdim, ..., old_pdim, old_pdim, ...).

        Return:
            fidelity (float) : approximation accuracy as fidelity
        """

        # apexをtidx[0]に合わせる
        if self.apex is not None:
            if tidx[0] < self.apex:
                for _ in range(self.apex - tidx[0]):
                    self.move_left_canonical()
            elif tidx[0] > self.apex:
                for _ in range(tidx[0] - self.apex):
                    self.move_right_canonical()
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1] - tidx[0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1] - tidx[i] != 1 or not is_direction_right and tidx[i+1] - tidx[i] != -1:
                raise ValueError("gate must be applied in sequential to MPO")
        
        gate = tn.Node(gtensor)
        for i in range(len(tidx)):
            self.nodes[tidx[i]][0] ^ gate[i+2*len(tidx)]
            self.nodes[tidx[i]][1] ^ gate[i+3*len(tidx)]

        node_edges = []
        for i in range(len(tidx)):
            node_edges.append(gate[i])
            gate[i].set_name(f"edge {tidx[i]}")
        for i in range(len(tidx)):
            node_edges.append(gate[i+len(tidx)])
            gate[i].set_name(f"edge {tidx[i]+self.n}")
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
            U, s, Vh, trun_s = tn.split_node_full_svd(tmp, left_edges, right_edges, self.truncate_dim, self.threshold_err)
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
        
        self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / self.calc_trace().flatten()[0]
        
        return total_fidelity

    def sample(self, seed=0):
        """ sample from mpo
        """
        #for _ in range(self.apex, 0, -1):
        #    self.move_left_canonical()

        np.random.seed(seed)

        output = []
        left_tensor = np.array([1])
        #for i in range(self.n-1):
        #    left_tensor.append(oe.contract("aacd,c->d", self.tensors[i], left_tensor[i]))
        right_tensor = [np.array([1])]
        for i in range(self.n-1, 0, -1):
            right_tensor.append(oe.contract("aacd,d->c", self.nodes[i].tensor, right_tensor[self.n-1-i]))
        right_tensor = right_tensor[::-1]
        zero = np.array([1, 0])
        one = np.array([0, 1])
        for i in range(self.n):
            prob_matrix = oe.contract("abcd,c,d->ab", self.nodes[i].tensor, left_tensor, right_tensor[i])
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
                left_tensor = oe.contract("abcd,a,b,c->d", self.nodes[i].tensor, zero.conj(), zero, left_tensor)
            else:
                output.append(1)
                left_tensor = oe.contract("abcd,a,b,c->d", self.nodes[i].tensor, one.conj(), one, left_tensor)
        
        return np.array(output)

    def calc_trace(self):
        left_tensor = oe.contract("aacd->cd", self.nodes[0].tensor)
        for i in range(1, self.n):
            left_tensor = oe.contract("ec,aacd->ed", left_tensor, self.nodes[i].tensor)
        return left_tensor

    def amplitude(self, tensors):
        """Caluculate one amplitude of MPO

        Args:
            tensors (List[np.array]) : the amplitude tensor, (0-phys,...,(n-1)-phys,0-conj,...,(n-1)-conj)
        
        Returns:
            np.array : result amplitude
        """

        left_tensor = oe.contract("abcd,a,b->cd", self.nodes[0].tensor, tensors[0], tensors[self.n])
        for i in range(1, self.n):
            left_tensor = oe.contract("ec,abcd,a,b->ed", left_tensor, self.nodes[i].tensor, tensors[i], tensors[self.n+i])
        return left_tensor

    def apply_MPO(self, tidx, mpo, is_truncate=False, is_normalize=False):
        """ apply MPO

        Args:
            tidx (List[int]) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
        """

        if is_truncate:
            raise NotImplementedError
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1] - tidx[0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1] - tidx[i] != 1 or not is_direction_right and tidx[i+1] - tidx[i] != -1:
                raise ValueError("mpo must be applied in sequential to MPO")
        
        total_fidelity = 1.0

        node_tensors = []

        # contract
        for i, node in enumerate(mpo.nodes):
            if i == 0:
                if is_direction_right:
                    if node[2].dimension != 1 and not self.nodes[tidx[i]][2].is_dangling():
                        raise ValueError("MPO has non-dim1 dangling edge at the first edge")
                else:
                    if node[2].dimension != 1 and not self.nodes[tidx[i]][3].is_dangling():
                        raise ValueError("MPO has non-dim1 dangling edge at the first edge")
            elif i == len(tidx) - 1:
                if is_direction_right:
                    if node[3].dimension != 1 and not self.nodes[tidx[i]][3].is_dangling():
                        raise ValueError("MPO has non-dim1 dangling edge at the final edge")
                else:
                    if node[3].dimension != 1 and not self.nodes[tidx[i]][2].is_dangling():
                        raise ValueError("MPO has non-dim1 dangling edge at the final edge")

            if is_direction_right:
                phys_dim = node[0].dimension
                conj_dim = self.nodes[tidx[i]][1].dimension
                left_dim = node[2].dimension * self.nodes[tidx[i]][2].dimension
                right_dim = node[3].dimension * self.nodes[tidx[i]][3].dimension
                node_tensors.append(oe.contract("abcd,befg->aefcgd",node.tensor,self.nodes[tidx[i]].tensor).reshape(phys_dim, conj_dim, left_dim, right_dim))
            else:
                phys_dim = node[0].dimension
                conj_dim = self.nodes[tidx[i]][1].dimension
                left_dim = node[3].dimension * self.nodes[tidx[i]][2].dimension
                right_dim = node[2].dimension * self.nodes[tidx[i]][3].dimension
                node_tensors.append(oe.contract("abcd,befg->aefdgc",node.tensor,self.nodes[tidx[i]].tensor).reshape(phys_dim, conj_dim, left_dim, right_dim))

        for i, t in enumerate(tidx):
            self.nodes[t].tensor = node_tensors[i]

        if is_normalize:
            trace = self.calc_trace() # must be 2D array
            if trace.shape[0] != 1:
                if trace.shape[0] == 4:
                    # connect with bell-pair
                    bell = np.array([1, 0, 0, 1]) / 2
                    trace = oe.contract("ab,a->b", trace, bell)
                else:
                    print("Error! trace of the MPO seems to be strange (cannnot calculated)")
            else:
                trace = trace[0]
            if trace.shape[0] != 1:
                if trace.shape[0] == 4:
                    # connect with bell-pair
                    bell = np.array([1, 0, 0, 1]) / 2
                    trace = oe.contract("a,a", trace, bell)
                else:
                    print("Error! trace of the MPO seems to be strange (cannnot calculated)")
            else:
                trace = trace[0]

            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / trace
        
        return total_fidelity
    
    def apply_MPO_as_CPTP(self, tidx, mpo, is_truncate=False, is_normalize=False, is_dangling_final=False):
        """ apply MPO as CPTP map

        Args:
            tidx (List[int]) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork. Note that the dimension of first dangling edge must be one
            is_truncate (bool) : truncate via canonical form or not
            is_normalize (bool) : normalize the final state or not
            is_dangling_final (bool) : absorb final dangling edge of mpo or not
        """

        if is_truncate:
            raise NotImplementedError
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1] - tidx[0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1] - tidx[i] != 1 or not is_direction_right and tidx[i+1] - tidx[i] != -1:
                raise ValueError("mpo must be applied in sequential to MPO")
        
        total_fidelity = 1.0

        node_tensors = []

        # contract
        for i, node in enumerate(mpo.nodes):
            if i == 0:
                if is_direction_right:
                    if node[2].dimension != 1:
                        raise ValueError("MPO has non-dim1 dangling edge at the first edge")
                else:
                    if node[2].dimension != 1:
                        raise ValueError("MPO has non-dim1 dangling edge at the first edge")
            elif i == len(tidx) - 1 and is_dangling_final:
                if is_direction_right:
                    if node[3].dimension != 1 and not self.nodes[tidx[i]][3].is_dangling():
                        raise ValueError("MPO has non-dim1 dangling edge at the final edge")
                else:
                    if node[3].dimension != 1 and not self.nodes[tidx[i]][2].is_dangling():
                        raise ValueError("MPO has non-dim1 dangling edge at the final edge")

            if i != len(tidx) - 1 or is_dangling_final:
                if is_direction_right:
                    phys_dim = conj_dim = node[0].dimension
                    left_dim = node[2].dimension * node[2].dimension * self.nodes[tidx[i]][2].dimension
                    right_dim = node[3].dimension * node[3].dimension * self.nodes[tidx[i]][3].dimension
                    node_tensors.append(oe.contract("abcd,befg,heij->ahfcigdj",node.tensor,self.nodes[tidx[i]].tensor,node.tensor.conj()).reshape(phys_dim, conj_dim, left_dim, right_dim))
                else:
                    phys_dim = node[0].dimension
                    conj_dim = self.nodes[tidx[i]][1].dimension
                    left_dim = node[3].dimension * node[3].dimension * self.nodes[tidx[i]][2].dimension
                    right_dim = node[2].dimension * node[2].dimension * self.nodes[tidx[i]][3].dimension
                    node_tensors.append(oe.contract("abdc,befg,heji->ahfcigdj",node.tensor,self.nodes[tidx[i]].tensor,node.tensor.conj()).reshape(phys_dim, conj_dim, left_dim, right_dim))
            else:
                if is_direction_right:
                    phys_dim = conj_dim = node[0].dimension
                    left_dim = node[2].dimension * node[2].dimension * self.nodes[tidx[i]][2].dimension
                    right_dim = self.nodes[tidx[i]][3].dimension
                    node_tensors.append(oe.contract("abcd,befg,heid->ahfcig",node.tensor,self.nodes[tidx[i]].tensor,node.tensor.conj()).reshape(phys_dim, conj_dim, left_dim, right_dim))
                else:
                    phys_dim = node[0].dimension
                    conj_dim = self.nodes[tidx[i]][1].dimension
                    left_dim = self.nodes[tidx[i]][2].dimension
                    right_dim = node[2].dimension * node[2].dimension * self.nodes[tidx[i]][3].dimension
                    node_tensors.append(oe.contract("abdc,befg,hejc->ahfgdj",node.tensor,self.nodes[tidx[i]].tensor,node.tensor.conj()).reshape(phys_dim, conj_dim, left_dim, right_dim))


        for i, t in enumerate(tidx):
            self.nodes[t].tensor = node_tensors[i]

        if is_normalize:
            trace = self.calc_trace() # must be 2D array
            if trace.shape[0] != 1:
                if trace.shape[0] == 4:
                    # connect with bell-pair
                    # bell = np.array([1, 0, 0, 1]) / 2
                    trace = oe.contract("ab,a->b", trace, bell)
                else:
                    print("Error! trace of the MPO seems to be strange (cannnot calculated)")
            else:
                trace = trace[0]
            if trace.shape[0] != 1:
                if trace.shape[0] == 4:
                    # connect with bell-pair
                    bell = np.array([1, 0, 0, 1]) / 2
                    trace = oe.contract("a,a", trace, bell)
                else:
                    print("Error! trace of the MPO seems to be strange (cannnot calculated)")
            else:
                trace = trace[0]

            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / trace
        
        return total_fidelity


    def move_right_canonical(self, threshold=1.0):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        l_edges = self.nodes[self.apex].get_all_edges()
        r_edges = self.nodes[self.apex+1].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [l_edges[0], l_edges[1], l_edges[2]], [l_edges[3]], max_truncation_err=1-threshold, relative=True)
        self.nodes[self.apex] = U.reorder_edges([l_edges[0], l_edges[1], l_edges[2], s[0]])
        self.nodes[self.apex+1] = tn.contractors.optimal([s, Vh, self.nodes[self.apex+1]], output_edge_order=[r_edges[0], r_edges[1], s[0], r_edges[3]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex+1].set_name(f"node {self.apex+1}")
        self.nodes[self.apex][3].set_name(f"edge {self.apex+2*self.n+1}")

        self.apex = self.apex + 1


    def move_left_canonical(self, threshold=1.0):
        """ move canonical apex to left
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        l_edges = self.nodes[self.apex-1].get_all_edges()
        r_edges = self.nodes[self.apex].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [r_edges[2]], [r_edges[0], r_edges[1], r_edges[3]], max_truncation_err=1-threshold, relative=True)
        self.nodes[self.apex] = Vh.reorder_edges([r_edges[0], r_edges[1], s[1], r_edges[3]])
        self.nodes[self.apex-1] = tn.contractors.optimal([self.nodes[self.apex-1], U, s], output_edge_order=[l_edges[0], l_edges[1], l_edges[2], s[1]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex-1].set_name(f"node {self.apex-1}")
        self.nodes[self.apex][2].set_name(f"edge {self.apex+2*self.n}")

        self.apex = self.apex - 1
