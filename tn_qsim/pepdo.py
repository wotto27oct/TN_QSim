import numpy as np
import opt_einsum as oe
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.utils import from_tn_to_quimb

class PEPDO(TensorNetwork):
    """class of PEPDO

    physical bond: 0, 1, ..., n-1
    inner bond: n, n+1, ..., 2n-1
    vertical virtual bond: 2n, 2n+1, ..., 2n+(height+1)-1, 2n+(height1), ..., 2n+(height+1)*width-1
    horizontal virtual bond: 2n+(height+1)*width, ..., 2n+(height+1)*width+(width+1)-1, ..., 2n+(height+1)*height+(height+1)*width-1

    edge index order for each node: 0(physical) 1(up) 2(right) 3(down) 4(left) 5(inner)

    Attributes:
        height (int) : PEPDO height
        width (int) : PEPDO width
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
        truncate_dim (int) : truncation dim of virtual bond, default None
        threthold_err (float) : the err threthold of singular values we keep
    """

    def __init__(self, tensors, height, width, truncate_dim=None, threthold_err=None, bmps_truncate_dim=None):
        self.n = len(tensors)
        self.height = height
        self.width = width
        self.path = None
        edge_info = []
        buff =2*self.n + (self.height+1)*self.width
        for h in range(self.height):
            for w in range(self.width):
                i = h*self.width + w
                edge_info.append([i, 2*self.n+w*(self.height+1)+h, buff+h*(self.width+1)+w+1, 2*self.n+w*(self.height+1)+h+1, buff+h*(self.width+1)+w, i+self.n])
        super().__init__(edge_info, tensors)
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err
        self.bmps_truncate_dim = bmps_truncate_dim
        self.tree, self.trace_tree = None, None


    @property
    def vertical_virtual_dims(self):
        virtual_dims = []
        for w in range(self.width):
            w_virtual_dims = [self.nodes[w].get_dimension(1)]
            for h in range(self.height):
                w_virtual_dims.append(self.nodes[w+h*self.width].get_dimension(3))
            virtual_dims.append(w_virtual_dims)
        return virtual_dims


    @property
    def horizontal_virtual_dims(self):
        virtual_dims = []
        for h in range(self.height):
            h_virtual_dims = [self.nodes[self.width].get_dimension(4)]
            for w in range(self.width):
                h_virtual_dims.append(self.nodes[w+h*self.width].get_dimension(2))
            virtual_dims.append(h_virtual_dims)
        return virtual_dims


    @property
    def inner_dims(self):
        inner_dims = []
        for i in range(self.n):
            inner_dims.append(self.nodes[i].get_dimension(5))
        return inner_dims


    def contract(self, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """contract PEPDO and generate full density operator

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            tree (ctg.ContractionTree) : the contraction tree
            path (list of tuple of int) : the contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))
        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            if cp_nodes[i].get_dimension(5) != 1:
                tn.connect(cp_nodes[i][5], cp_nodes[i+self.n][5])

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = [node for node in cp_nodes]

        for i in range(2*self.n):
            output_edge_order.append(cp_nodes[i][0])
        
        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)

    
    def prepare_trace(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))
        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            if cp_nodes[i].get_dimension(5) != 1:
                tn.connect(cp_nodes[i][5], cp_nodes[i+self.n][5])
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)
        node_list = [node for node in cp_nodes]

        return node_list, output_edge_order

    
    def calc_trace(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS"):
        """contract PEPDO and generate trace of full density operator

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
            node_list, output_edge_order = self.prepare_trace()
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)
        
        return self.contract_tree_by_quimb(tn, algorithm=algorithm, tree=tree, output_inds=output_inds, target_size=target_size, gpu=gpu, thread=thread, seq=seq)


    def find_trace_tree(self, algorithm=None, seq="ADCRS", visualize=False):
        """find contraction tree of the trace of the PEPDO

        Args:
            algorithm : the algorithm to find contraction path
            memory_limit : the maximum sp cost in contraction path
            visualize (bool) : if visualize whole contraction process
        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : total temporal cost
            max_sp_cost (int) : max spatial cost
        """

        node_list, output_edge_order = self.prepare_trace()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq, visualize)

        return tn, tree

    
    def __clear_dangling(self, cp_nodes):
        output_edge_order = []
        def clear_dangling(node_idx, dangling_index):
            one = tn.Node(np.array([1]))
            tn.connect(cp_nodes[node_idx][dangling_index], one[0])
            edge_order = []
            for i in range(len(cp_nodes[node_idx].edges)):
                if i != dangling_index:
                    edge_order.append(cp_nodes[node_idx][i])
            cp_nodes[node_idx] = tn.contractors.auto([cp_nodes[node_idx], one], edge_order)

        # 5,4,3,2,1の順に消す
        for i in range(self.n):
            if cp_nodes[i].get_dimension(5) == 1:
                clear_dangling(i, 5)
                clear_dangling(i+self.n, 5)
        # 表，裏
        for i in range(2):
            for h in range(self.height):
                if cp_nodes[i*self.n+h*self.width].get_dimension(4) == 1:
                    clear_dangling(i*self.n+h*self.width, 4)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+h*self.width][4])
            for w in range(self.width):
                if cp_nodes[i*self.n+self.width*(self.height-1)+w].get_dimension(3) == 1:
                    clear_dangling(i*self.n+self.width*(self.height-1)+w, 3)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+self.width*(self.height-1)+w][3])
            for h in range(self.height):
                if cp_nodes[i*self.n+(h+1)*self.width-1].get_dimension(2) == 1:
                    clear_dangling(i*self.n+(h+1)*self.width-1, 2)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+(h+1)*self.width-1][2])
            for w in range(self.width):
                if cp_nodes[i*self.n+w].get_dimension(1) == 1:
                    clear_dangling(i*self.n+w, 1)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+w][1])

        return cp_nodes, output_edge_order

    
    def apply_MPO(self, tidx, mpo):
        """ apply MPO
        
        Args:
            tidx (list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
        """

        def return_dir(diff):
            if diff == -self.width:
                return 1
            elif diff == 1:
                return 2
            elif diff == self.width:
                return 3
            elif diff == -1:
                return 4
            else:
                raise ValueError("must be applied sequentially")

        edge_list = []
        node_list = []

        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node, self.nodes[tidx[0]]]
            node_edge_list = [node[0]] + [self.nodes[tidx[0]][j] for j in range(1, 6)] + [node[3]]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            tn.connect(node[1], self.nodes[tidx[0]][0])
            new_node = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)
            tn.flatten_edges([new_node[5], new_node[6]])
            node_list.append(new_node)
        else:
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    node_contract_list = [node, self.nodes[tidx[i]]]
                    node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 6)] + [node[3]]
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_contract_list.append(one)
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i]][0])

                    # calc direction  up:1 right:2 down:3 left:4
                    dir = return_dir(tidx[i] - tidx[i-1])

                    # split nodes of PEPS via QR
                    l_l_edges = [node_list[i-1][j] for j in range(0, 6) if j != dir]
                    l_r_edges = [node_list[i-1][dir]] + [node_list[i-1][6]]
                    lQ, lR = tn.split_node_qr(node_list[i-1], l_l_edges, l_r_edges, edge_name="qr_left")
                    qr_left_edge = lQ.get_edge("qr_left")
                    lQ = lQ.reorder_edges(l_l_edges + [qr_left_edge])
                    lR = lR.reorder_edges(l_r_edges + [qr_left_edge])
                    r_l_edges = [self.nodes[tidx[i]][0]] + [self.nodes[tidx[i]][(dir+1)%4+1]]
                    r_r_edges = [self.nodes[tidx[i]][j] for j in range(1, 6) if j != (dir+1)%4+1]
                    rR, rQ = tn.split_node_rq(self.nodes[tidx[i]], r_l_edges, r_r_edges, edge_name="qr_right")
                    qr_right_edge = rR.get_edge("qr_right")
                    rR = rR.reorder_edges(r_l_edges + [qr_right_edge])
                    rQ = rQ.reorder_edges(r_r_edges + [qr_right_edge])

                    # contract left_R, right_R, node
                    svd_node_edge_list = None
                    svd_node_list = [lR, rR, node]
                    svd_node_edge_list = [qr_left_edge, node[0], node[3], qr_right_edge]
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                    # split via SVD for truncation
                    U, s, Vh, _ = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], self.truncate_dim)
                    l_edge_order = [lQ.edges[i] for i in range(0, dir)] + [s[0]] + [lQ.edges[i] for i in range(dir, 5)]
                    node_list[i-1] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)
                    r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 4)] + [Vh[2]]
                    new_node = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                    if i == mpo.n - 1:
                        tn.flatten_edges([new_node[5], new_node[6]])
                    node_list.append(new_node)

                    """l_l_edges = [node_list[i-1][j] for j in range(0, 6) if j != dir]
                    l_r_edges = [node_list[i-1][dir]] + [node_list[i-1][6]]
                    lU, ls, lVh, _ = tn.split_node_full_svd(node_list[i-1], l_l_edges, l_r_edges)
                    r_l_edges = [self.nodes[tidx[i]][0]] + [self.nodes[tidx[i]][(dir+1)%4+1]]
                    r_r_edges = [self.nodes[tidx[i]][j] for j in range(1, 6) if j != (dir+1)%4+1]
                    rU, rs, rVh, _ = tn.split_node_full_svd(self.nodes[tidx[i]], r_l_edges, r_r_edges)
                    lU = lU.reorder_edges(l_l_edges + [ls[0]])
                    rVh = rVh.reorder_edges(r_r_edges + [rs[1]])
                    svd_node_edge_list = [ls[0], node[0], node[3], rs[1]]
                    svd_node = tn.contractors.optimal([ls, lVh, rU, rs, node], output_edge_order=svd_node_edge_list)
                    U, s, Vh, _ = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[1], svd_node[2], svd_node[3]], self.truncate_dim)
                    l_edge_order = [lU.edges[i] for i in range(0, dir)] + [s[0]] + [lU.edges[i] for i in range(dir, 5)]
                    node_list[i-1] = tn.contractors.optimal([lU, U], output_edge_order=l_edge_order)
                    r_edge_order = [Vh[1]] + [rVh.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rVh.edges[i] for i in range((dir+1)%4, 4)] + [Vh[2]]
                    new_node = tn.contractors.optimal([s, Vh, rVh], output_edge_order=r_edge_order)
                    if i == mpo.n - 1:
                        tn.flatten_edges([new_node[5], new_node[6]])
                    node_list.append(new_node)"""

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]

    
    def prepare_Gamma(self, trun_node_idx):
        trun_node_idx, op_node_idx = trun_node_idx[0], trun_node_idx[1]
        trun_edge_idx = 0
        op_edge_idx = 0
        if trun_node_idx - op_node_idx == self.width:
            trun_edge_idx = 1
            op_edge_idx = 3
        elif trun_node_idx - op_node_idx == -1:
            trun_edge_idx = 2
            op_edge_idx = 4
        elif trun_node_idx - op_node_idx == -self.width:
            trun_edge_idx = 3
            op_edge_idx = 1
        else:
            trun_edge_idx = 4
            op_edge_idx = 2
        
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))
        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            if cp_nodes[i].get_dimension(5) != 1:
                tn.connect(cp_nodes[i][5], cp_nodes[i+self.n][5])
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
        
        cp_nodes[trun_node_idx][trun_edge_idx].disconnect("i", "j")
        cp_nodes[trun_node_idx+self.n][trun_edge_idx].disconnect("I", "J")
        edge_i = cp_nodes[trun_node_idx][trun_edge_idx]
        edge_I = cp_nodes[trun_node_idx+self.n][trun_edge_idx]
        edge_j = cp_nodes[op_node_idx][op_edge_idx]
        edge_J = cp_nodes[op_node_idx+self.n][op_edge_idx]
        output_edge_order = [edge_i, edge_I, edge_j, edge_J]

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes, output_edge_order1 = self.__clear_dangling(cp_nodes)
        # crear all other output_edge
        for i in range(len(output_edge_order1)//2):
            tn.connect(output_edge_order1[i], output_edge_order1[i+len(output_edge_order1)//2])
        node_list = [node for node in cp_nodes]

        return trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order


    def find_Gamma_tree(self, trun_node_idx, algorithm=None, seq="ADCRS", visualize=False):
        """find contraction tree of Gamma

        Args:
            trun_node_idx (list ofint) : the node index connected to the target edge
            truncate_dim (int) : the target bond dimension
            trial (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"
        
        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)

        if self.nodes[trun_node_idx][trun_edge_idx].dimension == 1:
            return None, None

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)
        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq, visualize)

        return tn, tree


    def find_optimal_truncation(self, trun_node_idx, truncate_dim=None, threthold=None, trials=10, algorithm=None, tnq=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS", visualize=False):
        """truncate the specified index using FET method

        Args:
            trun_node_idx (list of int) : the node index that we truncate
            truncate_dim (int) : the target bond dimension
            threthold (float) : the truncation threthold
            trials (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"

        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)
        
        if truncate_dim is not None and self.nodes[trun_node_idx][trun_edge_idx].dimension <= truncate_dim:
            print("trun_dim already satisfied")
            return 1.0

        # includes tree == None case
        output_inds = None
        if tnq is None:
            tnq, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        Gamma = self.contract_tree_by_quimb(tnq, algorithm=algorithm, tree=tree, output_inds=output_inds, target_size=target_size, gpu=gpu, thread=thread, seq=seq)

        if truncate_dim is None:
            truncate_dim = 1
        U, Vh, Fid = None, None, 1.0
        nU, nVh, nFid = None, None, 1.0
        if threthold is not None:
            for cur_truncate_dim in range(Gamma.shape[0] - 1, truncate_dim-1, -1):
                nU, nVh, nFid = self.find_optimal_truncation_by_Gamma(Gamma, cur_truncate_dim, trials, visualize=visualize)
                if nFid < threthold:
                    truncate_dim = cur_truncate_dim + 1
                    break
                U, Vh, Fid = nU, nVh, nFid
        else:
            # must be some truncate_dim
            U, Vh, Fid = self.find_optimal_truncation_by_Gamma(Gamma, truncate_dim, trials, visualize=visualize)


        old_trun = self.nodes[trun_node_idx].tensor
        old_op = self.nodes[op_node_idx].tensor
    
        # if truncation is executed        
        if U is not None:
            Unode = tn.Node(U)
            Vhnode = tn.Node(Vh)
            tn.connect(Unode[1], Vhnode[0])

            left_edge, right_edge = self.nodes[trun_node_idx][trun_edge_idx].disconnect()
            if left_edge.node1 != self.nodes[trun_node_idx]:
                left_edge, right_edge = right_edge, left_edge
            op_node = self.nodes[op_node_idx]

            # connect self.node[trun_node_idx] and Unode
            tn.connect(left_edge, Unode[0])
            node_contract_list = [self.nodes[trun_node_idx], Unode]
            node_edge_list = []
            for i in range(6):
                if i == trun_edge_idx:
                    node_edge_list.append(Unode[1])
                else:
                    node_edge_list.append(self.nodes[trun_node_idx][i])
            self.nodes[trun_node_idx] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)

            # connect op_node and Vhnode
            tn.connect(Vhnode[1], right_edge)
            node_contract_list = [op_node, Vhnode]
            node_edge_list = []
            for i in range(6):
                if i == op_edge_idx:
                    node_edge_list.append(Vhnode[0])
                else:
                    node_edge_list.append(op_node[i])
            self.nodes[op_node_idx] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)

        print(f"truncated from {Gamma.shape[0]} to {truncate_dim}, Fidelity: {Fid}")
        return Fid