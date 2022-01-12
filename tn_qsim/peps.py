import numpy as np
from scipy.sparse.linalg.eigen.arpack.arpack import CNEUPD_ERRORS
import tensornetwork as tn
from tn_qsim.mpo import MPO
from tn_qsim.mps import MPS
from tn_qsim.general_tn import TensorNetwork

class PEPS(TensorNetwork):
    """class of PEPS

    physical bond: 0, 1, ..., n-1
    vertical virtual bond: n, n+1, ..., n+(height+1)-1, n+(height1), ..., n+(height+1)*width-1
    horizontal virtual bond: n+(height+1)*width, ..., n+(height+1)*width+(width+1)-1, ..., n + (height+1)*height + (height+1)*width-1

    edge index order for each node: 0(physical) 1(up) 2(right) 3(down) 4(left)

    Attributes:
        height (int) : PEPS height
        width (int) : PEPS width
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
        buff = self.n + (self.height+1)*self.width
        for h in range(self.height):
            for w in range(self.width):
                i = h*self.width + w
                edge_info.append([i, self.n+w*(self.height+1)+h, buff+h*(self.width+1)+w+1, self.n+w*(self.height+1)+h+1, buff+h*(self.width+1)+w])
        super().__init__(edge_info, tensors)
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err
        self.bmps_truncate_dim = bmps_truncate_dim


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


    def contract(self, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """contract PEPS and generate full state

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

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = [node for node in cp_nodes]

        """for i in range(self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)"""
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])

        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)

    
    def amplitude(self, tensors, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """contract amplitude with given product states (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = []

        # contract product state first
        for i in range(self.n):
            state = tn.Node(tensors[i].conj())
            tn.connect(cp_nodes[i][0], state[0])
            edge_order = [cp_nodes[i].edges[j] for j in range(1, len(cp_nodes[i].edges))]
            #cp_nodes[i] = tn.contractors.auto([cp_nodes[i], state], edge_order)
            node_list.append(tn.contractors.auto([cp_nodes[i], state], edge_order))
            cp_nodes[i].tensor = None
            state.tensor = None

        #node_list = [node for node in cp_nodes]

        result = self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)
        return result

    
    def amplitude_BMPS(self, tensors):
        """calculate amplitude with given product states (typically computational basis) using BMPS

        Args:
            tensor (list of np.array) : the amplitude index represented by the list of tensor
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        # contract product state first
        for i in range(self.n):
            state = tn.Node(tensors[i].conj())
            tn.connect(cp_nodes[i][0], state[0])
            edge_order = [cp_nodes[i].edges[j] for j in range(1, len(cp_nodes[i].edges))]
            cp_nodes[i] = tn.contractors.auto([cp_nodes[i], state], edge_order)

        # suppose the dimension of down below is 1
        mps_node = []
        for w in range(self.width):
            tensor = cp_nodes[(self.height-1)*self.width+w].tensor
            shape = tensor.shape
            if w == 0:
                mps_node.append(tensor.reshape(shape[0], 1, shape[1]))
            elif w == self.width - 1:
                mps_node.append(tensor.reshape(shape[0], shape[1], 1))
            else:
                mps_node.append(tensor.reshape(shape[0], shape[1], shape[2]).transpose(0,2,1))

        mps = MPS(mps_node, truncate_dim=self.bmps_truncate_dim)
        mps.canonicalization()
        #print(f"mps at {self.height-1}th layer:", mps.virtual_dims)
        
        total_fid = 1.0

        for h in range(self.height-2, -1, -1):
            #print(f"height{h}")
            mpo_node = []
            for w in range(self.width):
                tensor = cp_nodes[h*self.width+w].tensor
                shape = tensor.shape
                if h != 0:
                    if w == 0:
                        tensor = tensor.reshape(shape[0], shape[1], shape[2], 1)
                    elif w == self.width - 1:
                        tensor = tensor.reshape(shape[0], 1, shape[1], shape[2])
                elif h == 0:
                    if w == 0:
                        tensor = tensor.reshape(1, shape[0], shape[1], 1)
                    elif w == self.width - 1:
                        tensor = tensor.reshape(1, 1, shape[0], shape[1])
                    else:
                        tensor = tensor.reshape(1, shape[0], shape[1], shape[2])
                mpo_node.append(tensor.transpose(0,2,3,1))
            mpo = MPO(mpo_node)
            fid = mps.apply_MPO([i for i in range(self.width)], mpo, is_normalize=False)
            #print("bmps mps-dim", mps.virtual_dims)
            total_fid = total_fid * fid
            
        return mps.contract().flatten()[0]
    
    
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

        # 4,3,2,1の順に消す
        for h in range(self.height):
            if cp_nodes[h*self.width].get_dimension(4) == 1:
                clear_dangling(h*self.width, 4)
            else:
                output_edge_order.append(cp_nodes[h*self.width][1])
        for w in range(self.width):
            if cp_nodes[self.width*(self.height-1)+w].get_dimension(3) == 1:
                clear_dangling(self.width*(self.height-1)+w, 3)
            else:
                output_edge_order.append(cp_nodes[self.width*(self.height-1)+w][1])
        for h in range(self.height):
            if cp_nodes[(h+1)*self.width-1].get_dimension(2) == 1:
                clear_dangling((h+1)*self.width-1, 2)
            else:
                output_edge_order.append(cp_nodes[(h+1)*self.width-1][1])
        for w in range(self.width):
            if cp_nodes[w].get_dimension(1) == 1:
                clear_dangling(w, 1)
            else:
                output_edge_order.append(cp_nodes[w][1])

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

        for i, node in enumerate(mpo.nodes):
            node_contract_list = [node, self.nodes[tidx[i]]]
            node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 5)]
            if i == 0:
                one = tn.Node(np.array([1]))
                tn.connect(node[2], one[0])
                node_contract_list.append(one)
                if i != mpo.n - 1:
                    node_edge_list.append(node[3])
            if i == mpo.n - 1:
                one = tn.Node(np.array([1]))
                tn.connect(node[3], one[0])
                node_contract_list.append(one)
                if i != 0:
                    node_edge_list.append(node[2])
            if i != 0 and i != mpo.n-1:
                node_edge_list = node_edge_list + [node[2], node[3]]
            
            tn.connect(node[1], self.nodes[tidx[i]][0])
            node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
            #edge_list.append(node_list[-1].edges)
        
        for i in range(len(tidx)):
            edge_list.append(node_list[i].edges)
            

        for i in range(len(tidx)-1):
            dir = return_dir(tidx[i+1] - tidx[i])
            edge_list[i][dir] = tn.flatten_edges([edge_list[i][dir], edge_list[i][-1]])
            edge_list[i+1][(dir+1)%4+1] = edge_list[i][dir]
            edge_list[i].pop()
            if i != len(tidx)-2:
                edge_list[i+1].pop(-2)
            else:
                edge_list[i+1].pop()
        
        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i].reorder_edges(edge_list[i])

    
    def apply_MPO_with_truncation(self, tidx, mpo):
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
            node_edge_list = [node[0]] + [self.nodes[tidx[0]][j] for j in range(1, 5)]
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
                    node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 5)] + [node[3]]
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_contract_list.append(one)
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    node_list.append(tn.contractors.optimal(node_contract_list, output_edge_order=node_edge_list))
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i]][0])

                    # calc direction  up:1 right:2 down:3 left:4
                    dir = return_dir(tidx[i] - tidx[i-1])

                    # split nodes of PEPS via QR
                    l_l_edges = [node_list[i-1][j] for j in range(0, 5) if j != dir]
                    l_r_edges = [node_list[i-1][dir]] + [node_list[i-1][5]]
                    lQ, lR = tn.split_node_qr(node_list[i-1], l_l_edges, l_r_edges, edge_name="qr_left")
                    qr_left_edge = lQ.get_edge("qr_left")
                    lQ = lQ.reorder_edges(l_l_edges + [qr_left_edge])
                    lR = lR.reorder_edges(l_r_edges + [qr_left_edge])
                    r_l_edges = [self.nodes[tidx[i]][0]] + [self.nodes[tidx[i]][(dir+1)%4+1]]
                    r_r_edges = [self.nodes[tidx[i]][j] for j in range(1, 5) if j != (dir+1)%4+1]
                    rR, rQ = tn.split_node_rq(self.nodes[tidx[i]], r_l_edges, r_r_edges, edge_name="qr_right")
                    qr_right_edge = rR.get_edge("qr_right")
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
                    U, s, Vh, _ = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], self.truncate_dim)
                    l_edge_order = [lQ.edges[i] for i in range(0, dir)] + [s[0]] + [lQ.edges[i] for i in range(dir, 4)]
                    node_list[i-1] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)
                    if i == mpo.n - 1:
                        r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 3)]
                        node_list.append(tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order))
                    else:
                        r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 3)] + [Vh[2]]
                        node_list.append(tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order))

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]


    def find_Gamma_tree(self, trun_node_idx, trun_edge_idx, algorithm=None, memory_limit=None, visualize=False):
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"
        op_node_idx = 0
        op_edge_idx = 0
        if trun_edge_idx == 1:
            op_node_idx = trun_node_idx - self.width
            op_edge_idx = 3
        elif trun_edge_idx == 2:
            op_node_idx = trun_node_idx + 1
            op_edge_idx = 4
        elif trun_edge_idx == 3:
            op_node_idx = trun_node_idx + self.width
            op_edge_idx = 1
        else:
            op_node_idx = trun_node_idx - 1
            op_edge_idx = 2

        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))

        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
        
        cp_nodes[trun_node_idx][trun_edge_idx].disconnect("i", "j")
        cp_nodes[trun_node_idx+self.n][trun_edge_idx].disconnect("I", "J")
        edge_i = cp_nodes[trun_node_idx][trun_edge_idx]
        edge_I = cp_nodes[trun_node_idx+self.n][trun_edge_idx]
        edge_j = cp_nodes[op_node_idx][op_edge_idx]
        edge_J = cp_nodes[op_node_idx+self.n][op_edge_idx]
        output_edge_order = [edge_i, edge_I, edge_j, edge_J]

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes1, output_edge_order1 = self.__clear_dangling(cp_nodes[:self.n])
        cp_nodes2, output_edge_order2 = self.__clear_dangling(cp_nodes[self.n:])
        node_list = [node for node in cp_nodes1 + cp_nodes2]

        tree, cost, sp_cost = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit)
        return tree, cost, sp_cost

    def find_optimal_truncation(self, trun_node_idx, trun_edge_idx, truncate_dim=None, threthold=None, trials=10, algorithm=None, memory_limit=None, visualize=False):
        """truncate the specified index using FET method

        Args:
            trun_node_idx (int) : the node index connected to the target edge
            trun_edge_idx (int) : the target edge's index of the above node
            truncate_dim (int) : the target bond dimension
            trial (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"
        op_node_idx = 0
        op_edge_idx = 0
        if trun_edge_idx == 1:
            op_node_idx = trun_node_idx - self.width
            op_edge_idx = 3
        elif trun_edge_idx == 2:
            op_node_idx = trun_node_idx + 1
            op_edge_idx = 4
        elif trun_edge_idx == 3:
            op_node_idx = trun_node_idx + self.width
            op_edge_idx = 1
        else:
            op_node_idx = trun_node_idx - 1
            op_edge_idx = 2

        if truncate_dim is not None and self.nodes[trun_node_idx][trun_edge_idx].dimension <= truncate_dim:
            print("trun_dim already satisfied")
            return

        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))

        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
        
        cp_nodes[trun_node_idx][trun_edge_idx].disconnect("i", "j")
        cp_nodes[trun_node_idx+self.n][trun_edge_idx].disconnect("I", "J")
        edge_i = cp_nodes[trun_node_idx][trun_edge_idx]
        edge_I = cp_nodes[trun_node_idx+self.n][trun_edge_idx]
        edge_j = cp_nodes[op_node_idx][op_edge_idx]
        edge_J = cp_nodes[op_node_idx+self.n][op_edge_idx]
        output_edge_order = [edge_i, edge_I, edge_j, edge_J]

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes1, output_edge_order1 = self.__clear_dangling(cp_nodes[:self.n])
        cp_nodes2, output_edge_order2 = self.__clear_dangling(cp_nodes[self.n:])
        node_list = [node for node in cp_nodes1 + cp_nodes2]

        Gamma = self.contract_tree(node_list, output_edge_order, algorithm, memory_limit)
        # truncate while Fid < threthold
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
            for i in range(5):
                if i == trun_edge_idx:
                    node_edge_list.append(Unode[1])
                else:
                    node_edge_list.append(self.nodes[trun_node_idx][i])
            self.nodes[trun_node_idx] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)

            # connect op_node and Vhnode
            tn.connect(Vhnode[1], right_edge)
            node_contract_list = [op_node, Vhnode]
            node_edge_list = []
            for i in range(5):
                if i == op_edge_idx:
                    node_edge_list.append(Vhnode[0])
                else:
                    node_edge_list.append(op_node[i])
            self.nodes[op_node_idx] = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)

        print(f"truncated from {Gamma.shape[0]} to {truncate_dim}, Fidelity: {Fid}")
        return Fid