import numpy as np
import opt_einsum as oe
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.utils import from_tn_to_quimb

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
        threshold_err (float) : the err threshold of singular values we keep
    """

    def __init__(self, tensors, truncate_dim=None, threshold_err=None):
        self.n = len(tensors)
        edge_info = []
        for i in range(self.n):
            edge_info.append([i, self.n+i, self.n+i+1])
        super().__init__(edge_info, tensors)
        self.apex = None
        self.truncate_dim = truncate_dim
        self.threshold_err = threshold_err

    @property
    def virtual_dims(self):
        virtual_dims = [self.nodes[0].get_dimension(1)]
        for i in range(self.n):
            virtual_dims.append(self.nodes[i].get_dimension(2))
        return virtual_dims


    def canonicalization(self, threshold=1.0):
        """canonicalize MPO
        apex point is set to be self.0

        Args:
            threshold (float) : truncation threshold for svd
        """
        self.apex = 0
        for i in range(self.n-1):
            self.__move_right_canonical(threshold)
        for i in range(self.n-1):
            self.__move_left_canonical(threshold)

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


    def prepare_inner(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))
        output_edge_order = []

        def clear_dangling(node_idx, dangling_index):
            one = tn.Node(np.array([1]))
            tn.connect(cp_nodes[node_idx][dangling_index], one[0])
            edge_order = []
            for i in range(len(cp_nodes[node_idx].edges)):
                if i != dangling_index:
                    edge_order.append(cp_nodes[node_idx][i])
            cp_nodes[node_idx] = tn.contractors.auto([cp_nodes[node_idx], one], edge_order)

        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        clear_dangling(0, 1)
        clear_dangling(self.n, 1)
        clear_dangling(self.n-1, 2)
        clear_dangling(2*self.n-1, 2)
        node_list = [node for node in cp_nodes]

        return node_list, output_edge_order

    
    def calc_inner(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS"):
        """calc inner product of PEPS state

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            tree (ctg.ContractionTree) : the contraction tree
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            np.array: tensor after contraction
        """

        output_inds = None
        if tn is None:
            node_list, output_edge_order = self.prepare_inner()
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)
        
        return self.contract_tree_by_quimb(tn, algorithm=algorithm, tree=tree, output_inds=output_inds, target_size=target_size, gpu=gpu, thread=thread, seq=seq)


    def find_inner_tree(self, algorithm=None, seq="ADCRS", visualize=False):
        """find contraction path of inner product of PEPS state

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : total temporal cost
            max_sp_cost (int) : max spatial cost
        """

        node_list, output_edge_order = self.prepare_inner()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq, visualize)

        return tn, tree

    
    def visualize_inner_tree(self, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """find contraction path of inner product of PEPS state

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : total temporal cost
            max_sp_cost (int) : max spatial cost
        """

        node_list, output_edge_order = self.prepare_inner()

        if tree is None and path is None:
            raise ValueError("tree or path is needed for visualization")

        self.visualize_tree(tree, node_list, output_edge_order, path=path, visualize=visualize)
        return

    def calc_trace(self):
        left_tensor = oe.contract("abc,aBC->bBcC", self.nodes[0].tensor, self.nodes[0].tensor.conj())
        for i in range(1, self.n):
            left_tensor = oe.contract("dDbB,abc,aBC->dDcC", left_tensor, self.nodes[i].tensor, self.nodes[i].tensor.conj())
        left_dim = left_tensor.shape[0] ** 2
        return left_tensor.reshape(left_dim, -1)


    def amplitude(self, tensors):
        """Caluculate one amplitude of MPO

        Args:
            tensors (List[np.array]) : the amplitude tensor, (0-phys,...,(n-1)-phys,0-conj,...,(n-1)-conj)
        
        Returns:
            np.array : result amplitude
        """

        left_tensor = oe.contract("abc,a->bc", self.nodes[0].tensor, tensors[0])
        for i in range(1, self.n):
            left_tensor = oe.contract("eb,abc,a->ec", left_tensor, self.nodes[i].tensor, tensors[i])
        return left_tensor

    
    def prepare_foliation(self, cut_list):
        node_list, output_edge_order = self.prepare_inner()

        rho_edge_order1 = []
        rho_edge_order2 = []
        for node_idx1, edge_idx1, node_idx2, edge_idx2 in cut_list:
            if node_list[node_idx1][edge_idx1] != node_list[node_idx2][edge_idx2]:
                print("error! cut_list is not correct", node_idx1, edge_idx1)
            node_list[node_idx1][edge_idx1].disconnect()
            rho_edge_order1.append(node_list[node_idx1][edge_idx1])
            rho_edge_order2.append(node_list[node_idx2][edge_idx2])
        
        output_edge_order = rho_edge_order1 + rho_edge_order2

        return node_list, output_edge_order


    def find_calc_foliation(self, cut_list, algorithm=None, seq="ADCRS", visualize=False):
        """find calc_foliation contraction path by using quimb

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_foliation(cut_list)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def calc_foliation(self, cut_list=None, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """calc foliation of MERA state

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_foliation(cut_list)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)

    def prepare_gamma(self, bond_idx):
        node_list, output_edge_order = self.prepare_inner()

        if bond_idx == 0:
            edge_idx = 1
        else:
            edge_idx = 2

        # split by bond_idx
        node_list[bond_idx][edge_idx].disconnect("i", "j")
        node_list[bond_idx+self.n][edge_idx].disconnect("I", "J")
        edge_i = node_list[bond_idx][edge_idx]
        edge_I = node_list[bond_idx+self.n][edge_idx]
        edge_j = node_list[bond_idx+1][1]
        edge_J = node_list[bond_idx+1+self.n][1]
        output_edge_order = [edge_i, edge_I, edge_j, edge_J]

        return node_list, output_edge_order

    def calc_gamma(self, bond_idx, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS"):
        """calc gamma of MPS state for gauge fixing

        Args:
            bond_idx (int) : the index of bond, leftmost is 0, rightmost is n-1
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            tree (ctg.ContractionTree) : the contraction tree
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            np.array: tensor after contraction
        """

        output_inds = None
        if tn is None:
            node_list, output_edge_order = self.prepare_gamma(bond_idx)
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)
        
        return self.contract_tree_by_quimb(tn, algorithm=algorithm, tree=tree, output_inds=output_inds, target_size=target_size, gpu=gpu, thread=thread, seq=seq)

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
            U, s, Vh, trun_s = tn.split_node_full_svd(tmp, left_edges, right_edges, self.truncate_dim, self.threshold_err)
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


    def apply_MPO(self, tidx, mpo, is_truncate=False, is_normalize=False, is_return_history=False):
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
                raise ValueError("mpo must be applied in sequential to MPS")
        
        dir = 2 if is_direction_right else 1

        total_fidelity = 1.0
            
        edge_list = []
        node_list = []
        mps_list = []
        if len(tidx) == 1:
            node = mpo.nodes[0]
            if node[2].dimension != 1 and not self.nodes[tidx[0]][1].is_dangling():
                raise ValueError("MPO has non-dim1 dangling edge at the first edge")
            if node[3].dimension != 1 and not self.nodes[tidx[0]][2].is_dangling():
                raise ValueError("MPO has non-dim1 dangling edge at the final edge")
            node_contract_list = [node, self.nodes[tidx[0]]]
            node_edge_list = [node[0]] + [self.nodes[tidx[0]][j] for j in range(1, 3)]
            if node[2].dimension == 1:
                one = tn.Node(np.array([1]))
                tn.connect(node[2], one[0])
                node_contract_list.append(one)
            else:
                node_edge_list.append(node[2])
            if node[3].dimension == 1:
                one2 = tn.Node(np.array([1]))
                tn.connect(node[3], one2[0])
                node_contract_list.append(one2)
            else:
                node_edge_list.append(node[3])
            tn.connect(node[1], self.nodes[tidx[0]][0])
            new_node = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)
            if node[3].dimension != 1:
                # flatten right edges, 0,1,2,3,4->0,1,3,(2,4) or 0,1,2,3->0,1,(2,3)
                tn.flatten_edges([new_node[2], new_node[-1]])
            if node[2].dimension != 1:
                # flatten left edges, 0,1,2,3->0,3,(1,2)->0,(1,2),3
                tn.flatten_edges([new_node[1], new_node[2]])
                reorder_list = [new_node[i] for i in [0,2,1]]
                new_node.reorder_edges(reorder_list)
            node_list.append(new_node)
        else:
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    node_contract_list = [node, self.nodes[tidx[i]]]
                    node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 3)] + [node[3]]
                    if node[2].dimension == 1:
                        one = tn.Node(np.array([1]))
                        tn.connect(node[2], one[0])
                        node_contract_list.append(one)
                    else:
                        node_edge_list.append(node[2])
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    new_node = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)
                    if node[2].dimension != 1:
                        if is_direction_right:
                            # flatten left edges, 0, 1, 2, 3, 4 -> 0, 2, 3, (1,4) -> 0, (1,4), 2, 3
                            tn.flatten_edges([new_node[1], new_node[4]])
                            reorder_list = [new_node[i] for i in [0,3,1,2]]
                            new_node.reorder_edges(reorder_list)
                        else:
                            # flatten left edges, 0, 1, 2, 3, 4 -> 0, 1, 3, (2,4) -> 0, 1, (2,4), 3
                            tn.flatten_edges([new_node[2], new_node[4]])
                            reorder_list = [new_node[i] for i in [0,1,3,2]]
                            new_node.reorder_edges(reorder_list)
                    node_list.append(new_node)
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i]][0])

                    # split nodes of MPS via QR
                    l_l_edges = [node_list[i-1][0], node_list[i-1][dir%2+1]]
                    l_r_edges = [node_list[i-1][dir], node_list[i-1][3]]
                    lQ, lR = tn.split_node_qr(node_list[i-1], l_l_edges, l_r_edges, edge_name="qr_left")
                    qr_left_edge = lQ.get_edge("qr_left")
                    lQ = lQ.reorder_edges(l_l_edges + [qr_left_edge])
                    lR = lR.reorder_edges(l_r_edges + [qr_left_edge])
                    r_l_edges = [self.nodes[tidx[i]][0], self.nodes[tidx[i]][dir%2+1]]
                    r_r_edges = [self.nodes[tidx[i]][dir]]
                    rR, rQ = tn.split_node_rq(self.nodes[tidx[i]], r_l_edges, r_r_edges, edge_name="qr_right")
                    qr_right_edge = rQ.get_edge("qr_right")
                    rR = rR.reorder_edges(r_l_edges + [qr_right_edge])
                    rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                    # contract left_R, right_R, node
                    svd_node_edge_list = None
                    svd_node_list = [lR, rR, node]
                    if i == mpo.n - 1 and node[3].dimension == 1:
                        one = tn.Node(np.array([1]))
                        tn.connect(node[3], one[0])
                        svd_node_edge_list = [qr_left_edge, node[0], qr_right_edge]
                        svd_node_list.append(one)
                    else:
                        svd_node_edge_list = [qr_left_edge, node[0], node[3], qr_right_edge]
                    
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                    # split via SVD for truncation
                    if is_truncate:
                        if svd_node.tensor.shape[0] > 1:
                            U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], self.truncate_dim, self.threshold_err, relative=True)
                        else:
                            U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], self.truncate_dim, self.threshold_err)
                    else:
                        U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))])
                    
                    # calc fidelity for normalization
                    if len(s.tensor) != 0:
                        s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                        trun_s_sq = np.dot(trun_s, trun_s)
                        fidelity = s_sq / (s_sq + trun_s_sq)
                        total_fidelity *= fidelity

                    l_edge_order = [lQ.edges[i] for i in range(0, dir)] + [s[0]] + [lQ.edges[i] for i in range(dir, 2)]
                    node_list[i-1] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)
                    r_edge_order = None
                    if i == mpo.n - 1 and node[3].dimension == 1:
                        if dir == 2: # right
                            r_edge_order = [Vh[1]] + [s[0]] + [rQ.edges[0]]
                        else:
                            r_edge_order = [Vh[1]] + [rQ.edges[0]] + [s[0]]
                    else:
                        if dir == 2: # right
                            r_edge_order = [Vh[1]] + [s[0]] + [rQ.edges[0]] + [Vh[2]]
                        else:
                            r_edge_order = [Vh[1]] + [rQ.edges[0]] + [s[0]] + [Vh[2]]
                    new_node = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                    if i == mpo.n - 1 and node[3].dimension != 1:
                        if dir == 2: # right:
                            # flatten right edges, 0, 1, 2, 3 -> 0, 1, (2, 3)
                            tn.flatten_edges([new_node[2], new_node[3]])
                        else:
                            # flatten left edges, 0, 1, 2, 3 -> 0, 2, (1,3) -> 0, (1,3), 2
                            tn.flatten_edges([new_node[1], new_node[3]])
                            reorder_list = [new_node[i] for i in [0,2,1]]
                            new_node.reorder_edges(reorder_list)
                            
                    node_list.append(new_node)
                
                if is_return_history:
                    tmp_mps_list = tn.replicate_nodes(node_list + self.nodes[len(node_list):])
                    mps_list.append(tmp_mps_list)
                    
        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]

        if self.apex is not None:
            self.apex = tidx[-1]
        
        #if is_normalize:
        #    self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(total_fidelity)

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

            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(trace)
        
        if not is_return_history:
            return total_fidelity
        else:
            return total_fidelity, mps_list


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


    def __move_right_canonical(self, threshold=1.0):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        l_edges = self.nodes[self.apex].get_all_edges()
        r_edges = self.nodes[self.apex+1].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [l_edges[0], l_edges[1]], [l_edges[2]], max_truncation_err=1-threshold, relative=True)
        self.nodes[self.apex] = U.reorder_edges([l_edges[0], l_edges[1], s[0]])
        self.nodes[self.apex+1] = tn.contractors.optimal([s, Vh, self.nodes[self.apex+1]], output_edge_order=[r_edges[0], s[0], r_edges[2]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex+1].set_name(f"node {self.apex+1}")
        self.nodes[self.apex][2].set_name(f"edge {self.apex+self.n+1}")

        self.apex = self.apex + 1

    
    def __move_left_canonical(self, threshold=1.0):
        """ move canonical apex to right
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        l_edges = self.nodes[self.apex-1].get_all_edges()
        r_edges = self.nodes[self.apex].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [r_edges[1]], [r_edges[0], r_edges[2]], max_truncation_err=1-threshold, relative=True)
        self.nodes[self.apex] = Vh.reorder_edges([r_edges[0], s[1], r_edges[2]])
        self.nodes[self.apex-1] = tn.contractors.optimal([self.nodes[self.apex-1], U, s], output_edge_order=[l_edges[0], l_edges[1], s[1]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex-1].set_name(f"node {self.apex-1}")
        self.nodes[self.apex][1].set_name(f"edge {self.apex+self.n}")

        self.apex = self.apex - 1