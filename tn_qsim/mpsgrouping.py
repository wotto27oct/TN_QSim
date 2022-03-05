import numpy as np
import opt_einsum as oe
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.utils import from_tn_to_quimb

class MPSgrouping(TensorNetwork):
    """class of MPS

    physical bond: 0, 1, ..., qnum-1
    virtual bond: qnum, n+1, ..., qnum+n

    Attributes:
        n (int) : the number of tensors
        qnum (int) : the number of physical index
        apex (int) : apex point of canonical form
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
        truncate_dim (int) : truncation dim of virtual bond, default None
        threthold_err (float) : the err threthold of singular values we keep
    """

    def __init__(self, tensors, truncate_dim=None, threthold_err=None):
        self.n = len(tensors)
        self.qnum = 0
        for tensor in tensors:
            self.qnum += tensor.ndim - 2
        edge_info = []
        buff = 0
        for i in range(self.n):
            edge_info_list = [buff+j for j in range(tensors[i].ndim-2)] + [self.qnum+i, self.qnum+i+1]
            edge_info.append(edge_info_list)
            buff += tensors[i].ndim-2
        super().__init__(edge_info, tensors)
        self.apex = None
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err

    @property
    def virtual_dims(self):
        virtual_dims = [self.nodes[0].get_dimension(len(self.nodes[0].tensor.shape)-2)]
        for i in range(self.n):
            virtual_dims.append(self.nodes[i].get_dimension(len(self.nodes[0].tensor.shape)-1))
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

    
    def prepare_contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        output_edge_order = []
        # if right-left dim is not 1, remain them
        if cp_nodes[0].get_dimension(len(cp_nodes[0].tensor.shape)-2) != 1:
            output_edge_order.append(cp_nodes[0][-2])
        if cp_nodes[self.n-1].get_dimension(len(cp_nodes[self.n-1].tensor.shape)-1) != 1:
            output_edge_order.append(cp_nodes[self.n-1][-1])
        for i in range(self.n):
            for j in range(len(cp_nodes[i].tensor.shape)-2):
                output_edge_order.append(cp_nodes[i][j])
        node_list = [node for node in cp_nodes]
        return node_list, output_edge_order
    
    def calc_contract(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS"):
        """contract MPSgrouping and generate pure state

        Args:
            algorithm : the algorithm to find contraction path
            tn : tensortetwork
            tree (ctg.ContractionTree) : the contraction tree
            visualize (bool) : if visualize whole contraction process
        Returns:
            np.array: tensor after contraction
        """
        
        output_inds = None
        if tn is None:
            node_list, output_edge_order = self.prepare_contract()
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)
        
        return self.contract_tree_by_quimb(tn, algorithm=algorithm, tree=tree, output_inds=output_inds, target_size=target_size, gpu=gpu, thread=thread, seq=seq)


    def find_contract_tree(self, algorithm=None, seq="ADCRS", visualize=False):
        """find contraction tree of the contract of MPS

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : total temporal cost
            max_sp_cost (int) : max spatial cost
        """

        node_list, output_edge_order = self.prepare_contract()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq, visualize)

        return tn, tree


    def contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        output_edge_order = [cp_nodes[0][1], cp_nodes[self.n-1][2]]
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])
        return tn.contractors.auto(cp_nodes, output_edge_order=output_edge_order).tensor
        


    def apply_MPO(self, tidx, mpo, is_normalize=True):
        """ apply MPO

        Args:
            tidx (list of list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
        """

        # tidx[0] == [0, 2] then apply mpo to edge self.node[0][2]

        # apexをtidx[0][0]に合わせる
        if self.apex is not None:
            if tidx[0][0] < self.apex:
                for _ in range(self.apex - tidx[0][0]):
                    self.__move_left_canonical()
            elif tidx[0][0] > self.apex:
                for _ in range(tidx[0][0] - self.apex):
                    self.__move_right_canonical()
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1][0] - tidx[0][0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1][0] - tidx[i][0] != 1 or not is_direction_right and tidx[i+1][0] - tidx[i][0] != -1:
                raise ValueError("gate must be applied in sequential to MPS")
        
        #dir = 2 if is_direction_right else 1
        dir = -1 if is_direction_right else -2
        op_dir = -2 if is_direction_right else -1

        total_fidelity = 1.0
            
        edge_list = []
        node_list = []
        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node, self.nodes[tidx[0][0]]]
            node_edge_list = self.nodes[tidx[0][0]][:tidx[0][1]] + [node[0]] + self.nodes[tidx[0][0]][tidx[0][1]+1:]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            one2 = tn.Node(np.array([1]))
            tn.connect(node[3], one2[0])
            node_contract_list.append(one2)
            tn.connect(node[1], self.nodes[tidx[0][0]][tidx[0][1]])
            node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
        else:
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    node_contract_list = [node, self.nodes[tidx[i][0]]]
                    node_edge_list = self.nodes[tidx[i][0]][:tidx[i][1]] + [node[0]] + self.nodes[tidx[i][0]][tidx[i][1]+1:] + [node[3]]
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_contract_list.append(one)
                    tn.connect(node[1], self.nodes[tidx[i][0]][tidx[i][1]])
                    node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i][0]][tidx[i][1]])

                    l_l_edges = node_list[i-1][:-3] + [node_list[i-1][op_dir-1]]
                    l_r_edges = [node_list[i-1][dir-1], node_list[i-1][-1]]
                    lQ, lR = tn.split_node_qr(node_list[i-1], l_l_edges, l_r_edges, edge_name="qr_left")
                    qr_left_edge = lQ.get_edge("qr_left")
                    lQ = lQ.reorder_edges(l_l_edges + [qr_left_edge])
                    lR = lR.reorder_edges(l_r_edges + [qr_left_edge])
                    r_l_edges = [self.nodes[tidx[i][0]][tidx[i][1]], self.nodes[tidx[i][0]][op_dir]]
                    r_r_edges = self.nodes[tidx[i][0]][:tidx[i][1]] + self.nodes[tidx[i][0]][tidx[i][1]+1:-2] + [self.nodes[tidx[i][0]][dir]]
                    rR, rQ = tn.split_node_rq(self.nodes[tidx[i][0]], r_l_edges, r_r_edges, edge_name="qr_right")
                    qr_right_edge = rQ.get_edge("qr_right")
                    rR = rR.reorder_edges(r_l_edges + [qr_right_edge])
                    rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])
                    
                    # contract left_R, right_R, node
                    svd_node_edge_list = None
                    svd_node_list = [lR, rR, node]
                    if i == mpo.n - 1:
                        one = tn.Node(np.array([1]))
                        tn.connect(node[3], one[0])
                        svd_node_edge_list = [qr_left_edge, node[0], qr_right_edge]
                        svd_node_list.append(one)
                    else:
                        svd_node_edge_list = [qr_left_edge, node[0], node[3], qr_right_edge]
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                    # split via SVD for truncation
                    U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], self.truncate_dim)

                    # calc fidelity for normalization
                    s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                    trun_s_sq = np.dot(trun_s, trun_s)
                    fidelity = s_sq / (s_sq + trun_s_sq)
                    total_fidelity *= fidelity

                    l_edge_order = None
                    if is_direction_right:
                        l_edge_order = lQ.edges[:-1] + [s[0]]
                    else:
                        l_edge_order = lQ.edges[:-2] + [s[0]] + [lQ.edges[-2]]
                    node_list[i-1] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)

                    r_edge_order = None
                    if i == mpo.n - 1:
                        if is_direction_right:
                            r_edge_order = rQ.edges[:tidx[i][1]] + [Vh[1]] + rQ.edges[tidx[i][1]:-2] + [s[0]] + [rQ.edges[-2]]
                        else:
                            r_edge_order = rQ.edges[:tidx[i][1]] + [Vh[1]] + rQ.edges[tidx[i][1]:-2] + [rQ.edges[-2]] + [s[0]]
                    else:
                        if is_direction_right:
                            r_edge_order = rQ.edges[:tidx[i][1]] + [Vh[1]] + rQ.edges[tidx[i][1]:-2] + [s[0]] + [rQ.edges[-2]] + [Vh[2]]
                        else:
                            r_edge_order = rQ.edges[:tidx[i][1]] + [Vh[1]] + rQ.edges[tidx[i][1]:-2] + [rQ.edges[-2]] + [s[0]] + [Vh[2]]
                    node_list.append(tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order))
                    
        for i in range(len(tidx)):
            self.nodes[tidx[i][0]] = node_list[i]

        if self.apex is not None:
            self.apex = tidx[-1][0]
        
        if is_normalize:
            self.nodes[tidx[-1][0]].tensor = self.nodes[tidx[-1][0]].tensor / np.sqrt(total_fidelity)
        
        return total_fidelity


    def __move_right_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        l_edges = self.nodes[self.apex].get_all_edges()
        r_edges = self.nodes[self.apex+1].get_all_edges()
        l_U_edges = l_edges[:-1]
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], l_edges[:-1], [l_edges[-1]])
        self.nodes[self.apex] = U.reorder_edges(l_edges[:-1] + [s[0]])
        self.nodes[self.apex+1] = tn.contractors.optimal([s, Vh, self.nodes[self.apex+1]], output_edge_order=r_edges[:-2] + [s[0], r_edges[-1]])

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
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [r_edges[-2]], r_edges[:-2] + [r_edges[-1]])
        self.nodes[self.apex] = Vh.reorder_edges(r_edges[:-2] + [s[1], r_edges[-1]])
        self.nodes[self.apex-1] = tn.contractors.optimal([self.nodes[self.apex-1], U, s], output_edge_order=l_edges[:-1] + [s[1]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex-1].set_name(f"node {self.apex-1}")
        self.nodes[self.apex][1].set_name(f"edge {self.apex+self.n}")

        self.apex = self.apex - 1