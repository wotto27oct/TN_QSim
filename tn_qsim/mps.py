import numpy as np
import opt_einsum as oe
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetworks

class MPS(TensorNetwork):
    """class of MPS

    physical bond: 0, 1, ..., n-1
    virtual bond: n, n+1, ..., 2n

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
            edge_info.append([i, self.n+i, self.n+i+1])
        super().__init__(edge_info, tensors)
        self.apex = None
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err

    @property
    def virtual_dims(self):
        virtual_dims = [self.nodes[0].get_dimension(1)]
        for i in range(self.n):
            virtual_dims.append(self.nodes[i].get_dimension(2))
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

    def set_apex(self, apex):
        self.canonicalization()
        for i in range(apex):
            self.__move_right_canonical()


    def contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        output_edge_order = [cp_nodes[0][1], cp_nodes[self.n-1][2]]
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])
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
        for i in range(len(tidx)):
            self.nodes[tidx[i]][0] ^ gate[i+len(tidx)]

        node_edges = []
        for i in range(len(tidx)):
            node_edges.append(gate[i])
            gate[i].set_name(f"edge {tidx[i]}")
        if is_direction_right:
            node_edges.append(self.nodes[tidx[0]][1])
            node_edges.append(self.nodes[tidx[-1]][2])
        else:
            node_edges.append(self.nodes[tidx[0]][2])
            node_edges.append(self.nodes[tidx[-1]][1])

        tmp = tn.contractors.optimal([self.nodes[i] for i in tidx] + [gate], ignore_edge_order=True)
        inner_edge = node_edges[-2]

        total_fidelity = 1.0

        for i in range(len(tidx)-1):
            left_edges = []
            right_edges = []
            left_edges.append(node_edges[i])
            left_edges.append(inner_edge)
            for j in range(len(tidx)-1-i):
                right_edges.append(node_edges[i+j+1])
            right_edges.append(node_edges[-1])
            U, s, Vh, trun_s = tn.split_node_full_svd(tmp, left_edges, right_edges, self.truncate_dim, self.threthold_err)
            U_reshape_edges = [node_edges[i], inner_edge, s[0]] if is_direction_right else [node_edges[i], s[0], inner_edge]
            self.nodes[tidx[i]] = U.reorder_edges(U_reshape_edges)
            inner_edge = s[0]
            tmp = tn.contractors.optimal([s, Vh], ignore_edge_order=True)

            self.nodes[tidx[i]].set_name(f"node {tidx[i]}")
            if is_direction_right:
                self.nodes[tidx[i]][2].set_name(f"edge {tidx[i]+self.n+1}")
            else:
                self.nodes[tidx[i]][1].set_name(f"edge {tidx[i]+self.n}")
            
            fidelity = 1.0 - np.dot(trun_s, trun_s)
            total_fidelity *= fidelity

        
        U_reshape_edges = [node_edges[len(tidx)-1], inner_edge, node_edges[-1]] if is_direction_right else [node_edges[len(tidx)-1], node_edges[-1], inner_edge]
        self.nodes[tidx[-1]] = tmp.reorder_edges(U_reshape_edges)
        self.nodes[tidx[-1]].set_name(f"node {tidx[-1]}")

        if self.apex is not None:
            self.apex = tidx[-1]
        
        self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(total_fidelity)
        
        return total_fidelity


    def apply_MPO(self, tidx, mpo, is_normalize=True):
        """ apply MPO

        Args:
            tidx (list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
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
        
        dir = 2 if is_direction_right else 1

        total_fidelity = 1.0
            
        edge_list = []
        node_list = []
        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node, self.nodes[tidx[0]]]
            node_edge_list = [node[0]] + [self.nodes[tidx[0]][j] for j in range(1, 3)]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            one2 = tn.Node(np.array([1]))
            tn.connect(node[3], one2[0])
            node_contract_list.append(one2)
            tn.connect(node[1], self.nodes[tidx[0]][0])
            node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
        else:
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    node_contract_list = [node, self.nodes[tidx[i]]]
                    node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 3)] + [node[3]]
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_contract_list.append(one)
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    
                    svd_node_list = [node_list[i-1], self.nodes[tidx[i]], node]
                    svd_node_edge_list = None
                    if i == mpo.n - 1:
                        one = tn.Node(np.array([1]))
                        tn.connect(node[3], one[0])
                        svd_node_edge_list = [node_list[i-1][0], node_list[i-1][dir%2+1], node[0], self.nodes[tidx[i]][dir]]
                        svd_node_list.append(one)
                    else:
                        svd_node_edge_list = [node_list[i-1][0], node_list[i-1][dir%2+1], node[0], self.nodes[tidx[i]][dir], node[3]]
                    #print("contraction to svd_node")
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)
                    #print("contraction to svd_node done")
                    #print("bmps svd", svd_node.tensor.shape)
                    U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0], svd_node[1]], [svd_node[i] for i in range(2, len(svd_node.edges))], self.truncate_dim)
                    #print("bmps svd done")
                    s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                    trun_s_sq = np.dot(trun_s, trun_s)
                    fidelity = s_sq / (s_sq + trun_s_sq)
                    total_fidelity *= fidelity
                    l_edge_order = [svd_node_edge_list[i] for i in range(0, dir)] + [s[0]] + [svd_node_edge_list[i] for i in range(dir, 2)]
                    node_list[i-1] = U.reorder_edges(l_edge_order)
                    r_edge_order = None
                    if i == mpo.n - 1:
                        if dir == 2: # right
                            r_edge_order = [svd_node_edge_list[2]] + [s[0]] + [svd_node_edge_list[3]]
                        else:
                            r_edge_order = [svd_node_edge_list[2]] + [svd_node_edge_list[3]] + [s[0]]
                    else:
                        if dir == 2: # right
                            r_edge_order = [svd_node_edge_list[2]] + [s[0]] + [svd_node_edge_list[3]] + [svd_node_edge_list[4]]
                        else:
                            r_edge_order = [svd_node_edge_list[2]] + [svd_node_edge_list[3]] + [s[0]] + [svd_node_edge_list[4]]
                    node_list.append(tn.contractors.optimal([s, Vh], output_edge_order=r_edge_order))
                    

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]

        if self.apex is not None:
            self.apex = tidx[-1]
        
        if is_normalize:
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(total_fidelity)
        
        return total_fidelity


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
            prob_matrix = oe.contract("abc,b,dec,e->ad", self.nodes[i].tensor, left_tensor, self.nodes[i].tensor.conj(), left_tensor.conj())
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
                left_tensor = oe.contract("abc,a,b->c", self.nodes[i].tensor, zero.conj(), left_tensor)
            else:
                output.append(1)
                left_tensor = oe.contract("abc,a,b->c", self.nodes[i].tensor, one.conj(), left_tensor)
        
        return np.array(output)


    def __move_right_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        l_edges = self.nodes[self.apex].get_all_edges()
        r_edges = self.nodes[self.apex+1].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [l_edges[0], l_edges[1]], [l_edges[2]])
        self.nodes[self.apex] = U.reorder_edges([l_edges[0], l_edges[1], s[0]])
        self.nodes[self.apex+1] = tn.contractors.optimal([s, Vh, self.nodes[self.apex+1]], output_edge_order=[r_edges[0], s[0], r_edges[2]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex+1].set_name(f"node {self.apex+1}")
        self.nodes[self.apex][2].set_name(f"edge {self.apex+self.n+1}")

        self.apex = self.apex + 1

    
    def __move_left_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        l_edges = self.nodes[self.apex-1].get_all_edges()
        r_edges = self.nodes[self.apex].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [r_edges[1]], [r_edges[0], r_edges[2]])
        self.nodes[self.apex] = Vh.reorder_edges([r_edges[0], s[1], r_edges[2]])
        self.nodes[self.apex-1] = tn.contractors.optimal([self.nodes[self.apex-1], U, s], output_edge_order=[l_edges[0], l_edges[1], s[1]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex-1].set_name(f"node {self.apex-1}")
        self.nodes[self.apex][1].set_name(f"edge {self.apex+self.n}")

        self.apex = self.apex - 1