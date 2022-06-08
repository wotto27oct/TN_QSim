import numpy as np
import tensornetwork as tn
from tn_qsim.mpo import MPO
from tn_qsim.mps import MPS
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.utils import *
import opt_einsum as oe
import copy

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
        self.inner_tree = None


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

    
    def prepare_contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = [node for node in cp_nodes]

        """for i in range(self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)"""
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])

        return node_list, output_edge_order


    def find_contract(self, algorithm=None, seq="ADCRS", visualize=False):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        node_list, output_edge_order = self.prepare_contract()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

    
    def contract(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=""):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        if tn is None:
            node_list, output_edge_order = self.prepare_contract()
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)
    
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


    def prepare_amplitude(self, tensors):
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = []

        # contract product state first
        for i in range(self.n):
            # if tensors[i] is None, leave it open
            if tensors[i] is None:
                output_edge_order.append(cp_nodes[-(self.n-i)][0])
                node_list.append(cp_nodes[-(self.n-i)])
            else:
                state = tn.Node(tensors[i].conj())
                tn.connect(cp_nodes[-(self.n-i)][0], state[0])
                edge_order = [cp_nodes[-(self.n-i)].edges[j] for j in range(1, len(cp_nodes[-(self.n-i)].edges))]
                node_list.append(tn.contractors.auto([cp_nodes[-(self.n-i)], state], edge_order))
                cp_nodes[-(self.n-i)].tensor = None
                state.tensor = None

        return node_list, output_edge_order


    def find_amplitude_tree(self, tensors, algorithm=None, seq="ADCRS", visualize=False):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        node_list, output_edge_order = self.prepare_amplitude(tensors)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

    
    def amplitude(self, tensors, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=""):
        """contract amplitude with given product states by using quimb (typically computational basis)

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
        
        if tn is None:
            node_list, output_edge_order = self.prepare_amplitude(tensors)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)

    
    def amplitude_BMPS(self, tensors):
        """calculate amplitude with given product states (typically computational basis) using BMPS
        !!warning!! open qubits must be at the top row.

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
            if tensors[i] is not None:
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
                # In case right-bond dim is not 1
                if len(shape) == 3:
                    mps_node.append(tensor.reshape(shape[0], shape[1], shape[2]).transpose(0,2,1))
                else:
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
                    # if tensor is open or not
                    if tensors[w] is not None:
                        if w == 0:
                            tensor = tensor.reshape(1, shape[0], shape[1], 1)
                        elif w == self.width - 1:
                            tensor = tensor.reshape(1, 1, shape[0], shape[1])
                        else:
                            tensor = tensor.reshape(1, shape[0], shape[1], shape[2])
                    else:
                        # if tensor is open
                        if w == 0:
                            tensor = tensor.reshape(shape[0], shape[1], shape[2], 1)
                        elif w == self.width - 1:
                            tensor = tensor.reshape(shape[0], 1, shape[1], shape[2])
                        else:
                            tensor = tensor.reshape(shape[0], shape[1], shape[2], shape[3])

                mpo_node.append(tensor.transpose(0,2,3,1))
            mpo = MPO(mpo_node)
            fid = mps.apply_MPO([i for i in range(self.width)], mpo, is_normalize=True)
            #print("bmps mps-dim", mps.virtual_dims)
            total_fid = total_fid * fid
            print(f"fidelity: {fid}")
            print(f"total fidelity: {total_fid}")

        return mps.contract().flatten(), total_fid
        #return mps.contract().flatten()[0]

    
    def prepare_inner(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))
        output_edge_order = []

        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes1, output_edge_order1 = self.__clear_dangling(cp_nodes[:self.n])
        cp_nodes2, output_edge_order2 = self.__clear_dangling(cp_nodes[self.n:])
        node_list = [node for node in cp_nodes1 + cp_nodes2]
        output_edge_order = [edge for edge in output_edge_order1 + output_edge_order2]

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

    def calc_inner_by_BMPS(self, truncate_dim=None, threthold=None, visualize=False):
        # contract inner and physical dim
        peps_tensors = []
        for idx in range(self.n):
            shape = self.nodes[idx].tensor.shape
            tmp = oe.contract("abcde,aBCDE->bBcCdDeE",self.nodes[idx].tensor, self.nodes[idx].tensor.conj())
            tmp = tmp.reshape(shape[1]**2, shape[2]**2, shape[3]**2, shape[4]**2)
            peps_tensors.append(tmp)
        
        # suppose the dimension of down below (except for left or right edges) is 1
        mps_tensors = []
        for w in range(self.width):
            tensor = peps_tensors[(self.height-1)*self.width+w]
            shape = tensor.shape
            if w == 0:
                mps_tensors.append(tensor.reshape(shape[0],shape[1],shape[2]*shape[3]).transpose(0,2,1))
            elif w == self.width-1:
                mps_tensors.append(tensor.reshape(shape[0],shape[1]*shape[2],shape[3]).transpose(0,2,1))
            else:
                mps_tensors.append(tensor.reshape(shape[0],shape[1],shape[3]).transpose(0,2,1))

        mps = MPS(mps_tensors, truncate_dim=truncate_dim, threthold_err=1.0-threthold)
        mps.canonicalization()

        total_fid = 1.0
        mps_tensors_list = [mps.tensors]
        # boundary MPS
        for h in range(self.height-2,-1,-1):
            mpo_tensors = []
            for w in range(self.width):
                tensor = peps_tensors[h*self.width+w]
                shape = tensor.shape
                mpo_tensors.append(tensor.transpose(0,2,3,1))
            mpo = MPO(mpo_tensors)
            fid = mps.apply_MPO([i for i in range(self.width)], mpo, is_normalize=False)
            #print("bmps mps-dim", mps.virtual_dims)
            total_fid = total_fid * fid
            if visualize:
                #print(f"fidelity: {fid}")
                #print(f"total fidelity: {total_fid}")
                print(f"MPS virtual dims: {mps.virtual_dims}")

        if visualize:
            print(f"total fidelity: {total_fid}")

        return mps.contract().flatten(), total_fid
    
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
                output_edge_order.append(cp_nodes[h*self.width][4])
        for w in range(self.width):
            if cp_nodes[self.width*(self.height-1)+w].get_dimension(3) == 1:
                clear_dangling(self.width*(self.height-1)+w, 3)
            else:
                output_edge_order.append(cp_nodes[self.width*(self.height-1)+w][3])
        for h in range(self.height):
            if cp_nodes[(h+1)*self.width-1].get_dimension(2) == 1:
                clear_dangling((h+1)*self.width-1, 2)
            else:
                output_edge_order.append(cp_nodes[(h+1)*self.width-1][2])
        for w in range(self.width):
            if cp_nodes[w].get_dimension(1) == 1:
                clear_dangling(w, 1)
            else:
                output_edge_order.append(cp_nodes[w][1])

        return cp_nodes, output_edge_order

    def apply_MPO_with_truncation(self, tidx, mpo, truncate_dim=None, last_dir=None):
        return self.apply_MPO(tidx, mpo, truncate_dim=truncate_dim, last_dir=last_dir)
    
    def apply_MPO(self, tidx, mpo, truncate_dim=None, last_dir=None):
        """ apply MPO with simple update
        
        Args:
            tidx (list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
            truncate_dim (int) : truncation dim
        """
        if truncate_dim is None:
            truncate_dim = self.truncate_dim

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

        # not accurate
        total_fidelity = 1.0

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
                    if i == mpo.n - 1 and last_dir is None:
                        one = tn.Node(np.array([1]))
                        tn.connect(node[3], one[0])
                        svd_node_edge_list = [qr_left_edge, node[0], qr_right_edge]
                        svd_node_list.append(one)
                    else:
                        svd_node_edge_list = [qr_left_edge, node[0], node[3], qr_right_edge]
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)

                    # split via SVD for truncation
                    U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], truncate_dim)

                    # calc fidelity
                    s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                    trun_s_sq = np.dot(trun_s, trun_s)
                    fidelity = s_sq / (s_sq + trun_s_sq)
                    total_fidelity *= fidelity

                    # reorder and flatten edges
                    l_edge_order = [lQ.edges[i] for i in range(0, dir)] + [s[0]] + [lQ.edges[i] for i in range(dir, 4)]
                    node_list[i-1] = tn.contractors.optimal([lQ, U], output_edge_order=l_edge_order)
                    new_node = None
                    if i == mpo.n - 1 and last_dir is None:
                        r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 3)]
                        new_node = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                    else:
                        r_edge_order = [Vh[1]] + [rQ.edges[i] for i in range(0, (dir+1)%4)] + [s[0]] + [rQ.edges[i] for i in range((dir+1)%4, 3)] + [Vh[2]]
                        new_node = tn.contractors.optimal([s, Vh, rQ], output_edge_order=r_edge_order)
                        if i == mpo.n-1 and last_dir is not None:
                            tn.flatten_edges([new_node[last_dir], new_node[5]])
                            reorder_list = [new_node[i] for i in range(last_dir)] + [new_node[4]] + [new_node[i] for i in range(last_dir, 4)]
                            new_node.reorder_edges(reorder_list)

                    node_list.append(new_node)

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]

        return total_fidelity
    
    def __return_idx_for_FET(self, trun_node_idx):
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
        return trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx

    def prepare_Gamma(self, trun_node_idx):
        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx = self.__return_idx_for_FET(trun_node_idx)
        
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
    
    def calc_Gamma(self, trun_node_idx, algorithm=None, tn=None, tree=None, output_inds=None, gpu=True, thread=1, seq=""):
        """calc Gamma

        Args:
            trun_node_idx (int, int) : (trun_node_idx, op_node_idx)
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """
    
        if tn is None or output_inds is None:
            trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)
            tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq)

        return self.contract_tree_by_quimb(tn, tree=tree, output_inds=output_inds, gpu=gpu)

    def __apply_bond_matrix(self, trun_node_idx, trun_edge_idx, op_node_idx, op_edge_idx, U, Vh):
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

    def __contract_node_inner(self, idx):
        shape = self.nodes[idx].tensor.shape
        tmp = oe.contract("abcde,aBCDE->bBcCdDeE",self.nodes[idx].tensor, self.nodes[idx].tensor.conj())
        tmp = tmp.reshape(shape[1]**2, shape[2]**2, shape[3]**2, shape[4]**2)
        # contract dangling
        # top
        if idx < self.width:
            if shape[1] != 1:
                tmp = oe.contract("abcd,a->bcd",tmp,np.array([1,0,0,1])).reshape(1,shape[2]**2,shape[3]**2,shape[4]**2)
        # right
        if idx % self.width == self.width-1:
            if shape[2] != 1:
                tmp = oe.contract("abcd,b->acd",tmp,np.array([1,0,0,1])).reshape(shape[1]**2,1,shape[3]**2,shape[4]**2)
        # down
        if idx >= self.n-self.width:
            if shape[3] != 1:
                tmp = oe.contract("abcd,c->abd",tmp,np.array([1,0,0,1])).reshape(shape[1]**2,shape[2]**2,1,shape[4]**2)
        # left
        if idx % self.width == 0:
            if shape[4] != 1:
                tmp = oe.contract("abcd,d->abc",tmp,np.array([1,0,0,1])).reshape(shape[1]**2,shape[2]**2,shape[3]**2,1)
        return tmp

    def __create_down_BMPS(self, bmps_truncate_dim=None, bmps_threthold=None):
        # contract inner, physical and dangling dim
        total_fid = 1.0
        peps_tensors = []
        for idx in range(self.n):
            tmp = self.__contract_node_inner(idx)
            peps_tensors.append(tmp)
        
        # BMPS from down right
        mps_down_tensors = [np.array([1]).reshape(1,1,1) for _ in range(self.width)]
        mps_down = MPS(mps_down_tensors, truncate_dim=bmps_truncate_dim, threthold_err=1-bmps_threthold)
        mps_down.canonicalization()
        mps_down_list = []
        for h in range(self.height-1,-1,-1):
            mpo_tensors = []
            for w in range(self.width-1,-1,-1):
                tensor = peps_tensors[h*self.width+w]
                shape = tensor.shape
                mpo_tensors.append(tensor.transpose(0,2,1,3))
            mpo = MPO(mpo_tensors)
            fid = mps_down.apply_MPO([i for i in range(self.width)], mpo, is_normalize=False)
            mps_down_list.append(copy.deepcopy(mps_down))
            print("bmps mps-dim for vertical FET", mps_down.virtual_dims)
            total_fid = total_fid * fid
            print(f"fidelity: {fid}")
            print(f"total fidelity: {total_fid}")

        mps_down_list = mps_down_list[::-1]
        self.mps_down_list = mps_down_list
        self.bmps_fidelity = total_fid

        return mps_down_list, total_fid

    def __create_right_BMPS(self, bmps_truncate_dim=None, bmps_threthold=None):
        # contract inner, physical and dangling dim
        total_fid = 1.0
        peps_tensors = []
        for idx in range(self.n):
            tmp = self.__contract_node_inner(idx)
            peps_tensors.append(tmp)
        
        # BMPS from down right
        mps_right_tensors = [np.array([1]).reshape(1,1,1) for _ in range(self.height)]
        mps_right = MPS(mps_right_tensors, truncate_dim=bmps_truncate_dim, threthold_err=1-bmps_threthold)
        mps_right.canonicalization()
        mps_right_list = []
        for w in range(self.width-1,-1,-1):
            mpo_tensors = []
            for h in range(self.height-1,-1,-1):
                tensor = peps_tensors[h*self.width+w]
                shape = tensor.shape
                mpo_tensors.append(tensor.transpose(3,1,2,0))
            mpo = MPO(mpo_tensors)
            fid = mps_right.apply_MPO([i for i in range(self.height)], mpo, is_normalize=False)
            mps_right_list.append(copy.deepcopy(mps_right))
            print("bmps mps-dim for horizontal FET", mps_right.virtual_dims)
            total_fid = total_fid * fid
            print(f"fidelity: {fid}")
            print(f"total fidelity: {total_fid}")

        mps_right_list = mps_right_list[::-1]
        self.mps_right_list = mps_right_list
        self.bmps_fidelity = total_fid

        return mps_right_list, total_fid

    def calc_horizontal_Gamma_by_BMPS(self, trun_node_idx, op_node_idx, bmps_truncate_dim=None, bmps_threshold=None, visualize=False):
        # calc horizontal Gamma (i.e. trun_node_idx=0, op_node_idx=1 etc...)

        if trun_node_idx > op_node_idx:
            trun_node_idx, op_node_idx = op_node_idx, trun_node_idx
            opposite_flag = True
        else:
            opposite_flag = False

        wpos = trun_node_idx % self.width
        hpos = trun_node_idx // self.width

        peps_tensors = []
        for idx in range(self.n):
            tmp = self.__contract_node_inner(idx)
            peps_tensors.append(tmp)

        # vertical BMPS from left
        mps_left = self.calc_horizontal_BMPS(0, wpos+1, 0, self.height, bmps_truncate_dim, bmps_threshold, visualize)

        # vertical BMPS from right
        mps_right = self.calc_horizontal_BMPS(self.width-1, wpos, 0, self.width, bmps_truncate_dim, bmps_threshold, visualize)

        left_node = mps_left.nodes
        right_node = mps_right.nodes
        node_contract_list = []
        for i in range(self.height):
            node_contract_list.append(left_node[i])
            node_contract_list.append(right_node[i])
            if i == hpos:
                continue
            tn.connect(left_node[i][0], right_node[i][0])
        one = tn.Node(np.array([1]))
        tn.connect(left_node[0][1], one[0])
        node_contract_list.append(one)
        one = tn.Node(np.array([1]))
        tn.connect(right_node[0][1], one[0])
        node_contract_list.append(one)
        one = tn.Node(np.array([1]))
        tn.connect(left_node[self.height-1][2], one[0])
        node_contract_list.append(one)
        one = tn.Node(np.array([1]))
        tn.connect(right_node[self.height-1][2], one[0])
        node_contract_list.append(one)
        output_edge_order = [left_node[hpos][0], right_node[hpos][0]]
        Gamma = tn.contractors.auto(node_contract_list, output_edge_order=output_edge_order).tensor
        shape = Gamma.shape
        dim1 = int(np.sqrt(shape[0]))
        Gamma = Gamma.reshape(dim1, dim1, dim1, dim1)

        if opposite_flag:
            Gamma = Gamma.transpose(2,3,0,1)
        return Gamma

    def calc_vertical_Gamma_by_BMPS(self, trun_node_idx, op_node_idx, bmps_truncate_dim=None, bmps_threshold=None, visualize=False):
        # calc vertical Gamma (i.e. trun_node_idx=0, op_node_idx=width etc...)

        if trun_node_idx > op_node_idx:
            trun_node_idx, op_node_idx = op_node_idx, trun_node_idx
            opposite_flag = True
        else:
            opposite_flag = False

        wpos = trun_node_idx % self.width
        hpos = trun_node_idx // self.width

        peps_tensors = []
        for idx in range(self.n):
            tmp = self.__contract_node_inner(idx)
            peps_tensors.append(tmp)
        
        # vertical BMPS from top
        mps_top = self.calc_vertical_BMPS(0, hpos+1, 0, self.width, bmps_truncate_dim, bmps_threshold, visualize)

        # vertical BMPS from down
        mps_down = self.calc_vertical_BMPS(self.height-1, hpos, 0, self.width, bmps_truncate_dim, bmps_threshold, visualize)

        # contract ladder tensor network

        top_node = mps_top.nodes
        down_node = mps_down.nodes
        node_contract_list = []
        for i in range(self.width):
            node_contract_list.append(top_node[i])
            node_contract_list.append(down_node[i])
            if i == wpos:
                continue
            tn.connect(top_node[i][0], down_node[i][0])
        one = tn.Node(np.array([1]))
        tn.connect(top_node[0][1], one[0])
        node_contract_list.append(one)
        one = tn.Node(np.array([1]))
        tn.connect(down_node[0][1], one[0])
        node_contract_list.append(one)
        one = tn.Node(np.array([1]))
        tn.connect(top_node[self.width-1][2], one[0])
        node_contract_list.append(one)
        one = tn.Node(np.array([1]))
        tn.connect(down_node[self.width-1][2], one[0])
        node_contract_list.append(one)
        output_edge_order = [top_node[wpos][0], down_node[wpos][0]]
        Gamma = tn.contractors.auto(node_contract_list, output_edge_order=output_edge_order).tensor
        shape = Gamma.shape
        dim1 = int(np.sqrt(shape[0]))
        Gamma = Gamma.reshape(dim1, dim1, dim1, dim1)

        if opposite_flag:
            Gamma = Gamma.transpose(2,3,0,1)
        return Gamma
        
    def calc_Gamma_by_BMPS(self, trun_node_idx, bmps_truncate_dim=None, bmps_threthold=None, visualize=False):
        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx = self.__return_idx_for_FET(trun_node_idx)

        if trun_edge_idx == 2 or trun_edge_idx == 4:
            return self.calc_horizontal_Gamma_by_BMPS(trun_node_idx, op_node_idx, bmps_truncate_dim, bmps_threthold, visualize=visualize)
        else:
            return self.calc_vertical_Gamma_by_BMPS(trun_node_idx, op_node_idx, bmps_truncate_dim, bmps_threthold, visualize=visualize)

    def bond_truncate_by_Gamma(self, trun_node_idx, Gamma, sigma, xinv=None, yinv=None, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threshold=None, trials=50, visualize=False):
        """ calc bond truncation using Gamma and sigma and update PEPS
        
        Args:
            trun_node_idx (int, int) : trun_node_idx, op_node_idx
            Gamma, sigma (np.array) : matrix for FET
            min_truncate_dim, max_truncate_dim, truncate_buff (int) : info for truncate iteration
            threshold (float) : needed fidelity
            trials (int) : rep number for optimal truncation
        return:
            Fid (float) : fidelity after optimal truncation
        """
        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx = self.__return_idx_for_FET(trun_node_idx)
        U, S, Vh, Fid = None, None, None, 1.0
        truncate_dim = None
        if threshold is not None:
            for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
                if cur_truncate_dim == Gamma.shape[0]:
                    print("no truncation done")
                    U = None
                    break
                elif Gamma.shape[0] <= cur_truncate_dim:         
                    print("truncate dim already satistfied")
                    U = None
                    break
                for sd in range(10):
                    U, S, Vh, Fid, trace = calc_optimal_truncation(Gamma, sigma, cur_truncate_dim, (1-threshold)/10, trials, visualize=visualize)
                    truncate_dim = cur_truncate_dim
                    if Fid > threshold:
                        break
                if Fid > threshold:
                    break
                    
        # if truncation is executed        
        if U is not None:
            if visualize:
                Gamma = oe.contract("iIjJ,ip,qj,IP,QJ->pPqQ",Gamma,U,Vh,U.conj(),Vh.conj())
                sigma = S
                print("Fid from Gamma, S:", oe.contract("iIjJ,ij,IJ",Gamma,sigma,sigma))
                sig = np.diag(sigma)
                sig = sig / np.linalg.norm(sig)
                print("WTG coef:", sig)
            U = np.dot(U, S) / np.sqrt(trace)

            if xinv is not None:
                # in case fix_gauge
                U = np.dot(xinv, U)
                Vh = np.dot(Vh, yinv)

            self.__apply_bond_matrix(trun_node_idx, trun_edge_idx, op_node_idx, op_edge_idx, U, Vh)

            print(f"truncate dim: {truncate_dim}")
            print(f"fidelity: {Fid}")
            return Fid
        else:
            return 1.0

    def bond_truncate_by_Gamma_exact(self, trun_node_idx, algorithm, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threshold=None, try_fix_gauge=False, trials=50, visualize=False):
        """ calc bond truncation using exact Gamma and update PEPS
        
        Args:
            trun_node_idx (int, int) : trun_node_idx, op_node_idx
            min_truncate_dim, max_truncate_dim, truncate_buff (int) : info for truncate iteration
            threshold (float) : needed fidelity
            trials (int) : rep number for optimal truncation
        return:
            Fid (float) : fidelity after optimal truncation
        """
        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx = self.__return_idx_for_FET(trun_node_idx)
        if min_truncate_dim is not None and self.nodes[trun_node_idx][trun_edge_idx].dimension <= min_truncate_dim:
            print("truncate dim already satisfied")
            return 1.0

        Gamma = self.calc_Gamma((trun_node_idx, op_node_idx), algorithm)

        if try_fix_gauge:
            Gamma, sigma, xinv, yinv = fix_gauge(Gamma)
        else:
            sigma, xinv, yinv = np.eye(Gamma.shape[0]), None, None

        print("Fid from Gamma, S:", oe.contract("iIjJ,ij,IJ",Gamma,sigma,sigma))
        Fid = self.bond_truncate_by_Gamma((trun_node_idx, op_node_idx), Gamma, sigma, xinv, yinv, min_truncate_dim, max_truncate_dim, truncate_buff, threshold, trials, visualize)

        return Fid

    def bond_truncate_by_Gamma_BMPS(self, trun_node_idx, bmps_truncate_dim=None, bmps_threshold=None, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threshold=None, try_fix_gauge=False, trials=50, visualize=False):
        """ calc bond truncation using BMPS Gamma for each bond and update PEPS
        
        Args:
            trun_node_idx (int, int) : trun_node_idx, op_node_idx
            bmps_truncate_dim, bmps_threshold (int, float) : bmps setting, bigger is better
            min_truncate_dim, max_truncate_dim, truncate_buff (int) : info for truncate iteration
            threshold (float) : needed fidelity
            trials (int) : rep number for optimal truncation
        return:
            Fid (float) : fidelity after optimal truncation
        """
        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx = self.__return_idx_for_FET(trun_node_idx)
        if min_truncate_dim is not None and self.nodes[trun_node_idx][trun_edge_idx].dimension <= min_truncate_dim:
            print("truncate dim already satisfied")
            return 1.0

        Gamma = self.calc_Gamma_by_BMPS((trun_node_idx, op_node_idx), bmps_truncate_dim, bmps_threshold, visualize)

        if try_fix_gauge:
            Gamma, sigma, xinv, yinv = fix_gauge(Gamma)
        else:
            sigma, xinv, yinv = np.eye(Gamma.shape[0]), None, None

        print("Fid from Gamma, S:", oe.contract("iIjJ,ij,IJ",Gamma,sigma,sigma))
        Fid = self.bond_truncate_by_Gamma((trun_node_idx, op_node_idx), Gamma, sigma, xinv, yinv, min_truncate_dim, max_truncate_dim, truncate_buff, threshold, trials, visualize)

        return Fid
        
    def bond_truncate_by_Gamma_BMPS_old(self, bmps_truncate_dim=None, bmps_threthold=None, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threthold=None, trials=20, gpu=True, visualize=False):
        total_fid = 1.0
        for w in range(self.width-1):
            for h in range(self.height):
                self.regularize_PEPS()
                
                if visualize:
                    print(f"horizontal h:{h} w:{w}")
                    inner_val , fid = self.calc_inner_by_BMPS(threthold=bmps_threthold)
                    print(f"BMPS trace: {inner_val} fid:{fid}")
                    print("singularity:", self.calc_singularity())
                    if np.real_if_close(inner_val.item()) < 0.9999:
                        print("inner calc error happened!!", inner_val)

                
                Gamma = self.calc_Gamma_by_BMPS((h*self.width+w, h*self.width+w+1), bmps_truncate_dim, bmps_threthold, visualize)
                sigma = np.eye(Gamma.shape[0])

                U, S, Vh, Fid = None, None, None, 1.0
                truncate_dim = None
                if threthold is not None:
                    for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
                        if cur_truncate_dim == Gamma.shape[0]:
                            print("no truncation done")
                            U = None
                            break
                        elif Gamma.shape[0] <= cur_truncate_dim:         
                            print("truncate dim already satistfied")
                            U = None
                            break
                        for sd in range(10):
                            U, S, Vh, Fid, trace = calc_optimal_truncation(Gamma, sigma, cur_truncate_dim, trials, visualize=visualize)
                            truncate_dim = cur_truncate_dim
                            if Fid > threthold:
                                break
                        if Fid > threthold:
                            break
                            
                # if truncation is executed        
                if U is not None:
                    if visualize:
                        Gamma = oe.contract("iIjJ,ip,qj,IP,QJ->pPqQ",Gamma,U,Vh,U.conj(),Vh.conj())
                        sigma = S
                        print("Gamma after optimal truncation", Gamma.reshape(Gamma.shape[0]**2, -1)[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
                        print("Fid from Gamma, S:", oe.contract("iIjJ,ij,IJ",Gamma,sigma,sigma))
                        print("cycle entropy:", calc_cycle_entropy(Gamma, sigma))
                        sig = np.diag(sigma)
                        sig = sig / np.linalg.norm(sig)
                        print("WTG coef:", sig)
                    U = np.dot(U, S) / np.sqrt(trace)

                    trun_node_idx = h*self.width+w
                    trun_edge_idx = 2
                    op_node_idx = h*self.width+w+1
                    op_edge_idx = 4

                    self.__apply_bond_matrix(trun_node_idx, trun_edge_idx, op_node_idx, op_edge_idx, U, Vh)

                    print(f"truncate dim: {truncate_dim}")
                    total_fid = total_fid * Fid
                    print(f"fidelity: {Fid}")
                    print(f"total fidelity: {total_fid}")

        return total_fid
    
    def bond_truncate_by_BMPS(self, bmps_truncate_dim=None, bmps_threthold=None, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threthold=None, trials=20, gpu=True, is_calc_BMPS=True, is_fix_gauge=False, visualize=False):
        total_fid = 1.0
        mps_down_list, mps_right_list = None, None
        if is_calc_BMPS:
            mps_down_list, fid = self.__create_down_BMPS(bmps_truncate_dim, bmps_threthold)
            total_fid *= fid
        else:
            mps_down_list = self.mps_down_list

        # vertical FET from top left
        # BMPS from top left
        mps_top_tensors = [np.array([1]).reshape(1,1,1) for _ in range(self.width)]
        mps_top = MPS(mps_top_tensors, truncate_dim=bmps_truncate_dim, threthold_err=1-bmps_threthold)
        mps_top.canonicalization()

        #for h in range(0):
        for h in range(self.height-1):
            # create top MPS
            mpo_tensors = []
            for w in range(self.width):
                tmp = self.__contract_node_inner(h*self.width+w)
                mpo_tensors.append(tmp.transpose(2,0,3,1))
            mpo = MPO(mpo_tensors)
            fid = mps_top.apply_MPO([i for i in range(self.width)], mpo, is_normalize=False)
            total_fid = total_fid * fid
            for w in range(self.width):
                # create Gamma to execute FET for each verical edges
                if visualize:
                    print(f"vertical h:{h} w:{w}")
                    print("BMPS trace:", self.calc_inner_by_BMPS(threthold=bmps_threthold))
                    print("singularity:", self.calc_singularity())
                top_nodes = tn.replicate_nodes(mps_top.nodes)
                down_nodes = tn.replicate_nodes(mps_down_list[h+1].nodes)
                node_contract_list = []
                for i in range(self.width):
                    node_contract_list.append(top_nodes[i])
                    node_contract_list.append(down_nodes[self.width-1-i])
                    if i == w:
                        continue
                    tn.connect(top_nodes[i][0], down_nodes[self.width-1-i][0])
                one = tn.Node(np.array([1]))
                tn.connect(top_nodes[0][1], one[0])
                node_contract_list.append(one)
                one = tn.Node(np.array([1]))
                tn.connect(down_nodes[0][1], one[0])
                node_contract_list.append(one)
                one = tn.Node(np.array([1]))
                tn.connect(top_nodes[self.width-1][2], one[0])
                node_contract_list.append(one)
                one = tn.Node(np.array([1]))
                tn.connect(down_nodes[self.width-1][2], one[0])
                node_contract_list.append(one)
                output_edge_order = [top_nodes[w][0], down_nodes[self.width-1-w][0]]
                Gamma = tn.contractors.auto(node_contract_list, output_edge_order=output_edge_order).tensor
                shape = Gamma.shape
                dim1 = int(np.sqrt(shape[0]))
                dim2 = int(np.sqrt(shape[1]))
                Gamma = Gamma.reshape(dim1, dim1, dim2, dim2)
                sigma = np.eye(dim1)
                ori_Gamma = Gamma

                # fix gauge
                if is_fix_gauge:
                    Gamma, sigma, xinv, yinv = fix_gauge(Gamma, visualize=False)

                    gnorm, snorm, xnorm, ynorm = np.linalg.norm(Gamma), np.linalg.norm(sigma), np.linalg.norm(xinv), np.linalg.norm(yinv)
                    if gnorm > 1e5 or snorm > 1e5 or xnorm > 1e5 or ynorm > 1e5:
                        print(f"unstable fixing, {gnorm} {snorm} {xnorm} {ynorm}")
                        Gamma = ori_Gamma
                        sigma = np.eye(sigma.shape[0])
                        xinv = np.eye(xinv.shape[0])
                        yinv = np.eye(yinv.shape[0])

                    if visualize:
                        print("Gamma after gauge fixing", Gamma.reshape(Gamma.shape[0]**2, -1)[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
                        print("is_WTG:", is_WTG(Gamma, sigma))
                        print("cycle entropy:", calc_cycle_entropy(Gamma, sigma))
                        sig = np.diag(sigma)
                        sig = sig / np.linalg.norm(sig)
                        print("WTG coef:", sig)
                """Gamma, sigma = fix_gauge(Gamma, visualize=visualize)

                if visualize:
                    print("Gamma after gauge fixing", Gamma.reshape(Gamma.shape[0]**2, -1))
                    print("is_WTG:", is_WTG(Gamma, sigma))
                    print("cycle entropy:", calc_cycle_entropy(Gamma, sigma))
                    sig = np.diag(sigma)
                    sig = sig / np.linalg.norm(sig)
                    print("WTG coef:", sig)"""
                
                # sigma = np.eye(dim1)

                U, S, Vh, Fid = None, None, None, 1.0
                truncate_dim = None
                if threthold is not None:
                    for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
                        if cur_truncate_dim == Gamma.shape[0]:
                            print("no truncation done")
                            U = None
                            break
                        elif Gamma.shape[0] <= cur_truncate_dim:         
                            print("truncate dim already satistfied")
                            U = None
                            break
                        for sd in range(10):
                            U, S, Vh, Fid, trace = calc_optimal_truncation(Gamma, sigma, cur_truncate_dim, trials, visualize=visualize)
                            truncate_dim = cur_truncate_dim
                            if Fid > threthold:
                                break
                        if Fid > threthold:
                            break
                            
                # if truncation is executed        
                if U is not None:
                    Unorm, Snorm, Vhnorm = np.linalg.norm(U), np.linalg.norm(S), np.linalg.norm(Vh)
                    if Unorm > 1e3 or Snorm > 1e3 or Vhnorm > 1e3:
                        print(f"optimal truncation unstable, {Unorm}, {Snorm}, {Vhnorm}")
                        U = None

                if U is not None:
                    if visualize:
                        Gamma = oe.contract("iIjJ,ip,qj,IP,QJ->pPqQ",Gamma,U,Vh,U.conj(),Vh.conj())
                        sigma = S
                        print("Gamma after optimal truncation", Gamma.reshape(Gamma.shape[0]**2, -1)[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
                        print("is_WTG:", is_WTG(Gamma, sigma))
                        print("cycle entropy:", calc_cycle_entropy(Gamma, sigma))
                        sig = np.diag(sigma)
                        sig = sig / np.linalg.norm(sig)
                        print("WTG coef:", sig)
                    U = np.dot(U, S) / np.sqrt(trace)
                    """news_dim = S.shape[0]
                    Tmp = np.dot(np.dot(U, S), Vh)
                    newU, news, newVh = np.linalg.svd(Tmp)
                    news = news[:news_dim]
                    newU = newU[:, :news_dim]
                    newVh = newVh[:news_dim, :]
                    U = np.dot(newU, np.diag(np.sqrt(news))) / np.sqrt(trace)
                    Vh = np.dot(np.diag(np.sqrt(news)), newVh)"""

                    trun_node_idx = h*self.width+w
                    trun_edge_idx = 3
                    op_node_idx = (h+1)*self.width+w
                    op_edge_idx = 1

                    if is_fix_gauge:
                        U = np.dot(xinv, U)
                        Vh = np.dot(Vh, yinv)

                    self.__apply_bond_matrix(trun_node_idx, trun_edge_idx, op_node_idx, op_edge_idx, U, Vh)

                    print(f"truncate dim: {truncate_dim}")
                    total_fid = total_fid * Fid
                    print(f"fidelity: {Fid}")
                    print(f"total fidelity: {total_fid}")

                    # also for mps_top_nodes, mps_down_list
                    Utensor = oe.contract("ij,IJ->jJiI",U,U.conj()).reshape(U.shape[1]**2,-1,1,1)
                    mps_top.apply_MPO([w], MPO([Utensor]))

                    Vhtensor = oe.contract("ij,IJ->iIjJ",Vh,Vh.conj()).reshape(Vh.shape[0]**2,-1,1,1)
                    mps_down_list[h+1].apply_MPO([self.width-1-w], MPO([Vhtensor]))

        # horizontal FET from top left

        if is_calc_BMPS:
            mps_right_list, fid = self.__create_right_BMPS(bmps_truncate_dim, bmps_threthold)
            total_fid *= fid
        else:
            mps_right_list = self.mps_right_list

        # BMPS from top left
        mps_left_tensors = [np.array([1]).reshape(1,1,1) for _ in range(self.height)]
        mps_left = MPS(mps_left_tensors, truncate_dim=bmps_truncate_dim, threthold_err=1-bmps_threthold)
        mps_left.canonicalization()

        for w in range(self.width-1):
        #for w in range(0):
            # create top MPS
            mpo_tensors = []
            for h in range(self.height):
                tmp = self.__contract_node_inner(h*self.width+w)
                mpo_tensors.append(tmp.transpose(1,3,0,2))
            mpo = MPO(mpo_tensors)
            fid = mps_left.apply_MPO([i for i in range(self.height)], mpo, is_normalize=False)
            total_fid = total_fid * fid
            for h in range(self.height):
                if visualize:
                    print(f"horizontal h:{h} w:{w}")
                    inner_val , fid = self.calc_inner_by_BMPS(threthold=bmps_threthold)
                    print(f"BMPS trace: {inner_val} fid:{fid}")
                    print("singularity:", self.calc_singularity())
                    if np.real_if_close(inner_val.item()) < 0.9999:
                        print("inner calc error happened!!", inner_val)
                left_node = tn.replicate_nodes(mps_left.nodes)
                right_nodes = tn.replicate_nodes(mps_right_list[w+1].nodes)
                node_contract_list = []
                for i in range(self.height):
                    node_contract_list.append(left_node[i])
                    node_contract_list.append(right_nodes[self.height-1-i])
                    if i == h:
                        continue
                    tn.connect(left_node[i][0], right_nodes[self.height-1-i][0])
                one = tn.Node(np.array([1]))
                tn.connect(left_node[0][1], one[0])
                node_contract_list.append(one)
                one = tn.Node(np.array([1]))
                tn.connect(right_nodes[0][1], one[0])
                node_contract_list.append(one)
                one = tn.Node(np.array([1]))
                tn.connect(left_node[self.height-1][2], one[0])
                node_contract_list.append(one)
                one = tn.Node(np.array([1]))
                tn.connect(right_nodes[self.height-1][2], one[0])
                node_contract_list.append(one)
                output_edge_order = [left_node[h][0], right_nodes[self.height-1-h][0]]
                Gamma = tn.contractors.auto(node_contract_list, output_edge_order=output_edge_order).tensor
                shape = Gamma.shape
                dim1 = int(np.sqrt(shape[0]))
                dim2 = int(np.sqrt(shape[1]))
                Gamma = Gamma.reshape(dim1, dim1, dim2, dim2)
                sigma = np.eye(dim1)
                ori_Gamma = Gamma

                # fix gauge
                if is_fix_gauge:
                    Gamma, sigma, xinv, yinv = fix_gauge(Gamma, visualize=False)

                    gnorm, snorm, xnorm, ynorm = np.linalg.norm(Gamma), np.linalg.norm(sigma), np.linalg.norm(xinv), np.linalg.norm(yinv)
                    if gnorm > 1e5 or snorm > 1e5 or xnorm > 1e5 or ynorm > 1e5:
                        print(f"unstable fixing, {gnorm} {snorm} {xnorm} {ynorm}")
                        Gamma = ori_Gamma
                        sigma = np.eye(sigma.shape[0])
                        xinv = np.eye(xinv.shape[0])
                        yinv = np.eye(yinv.shape[0])

                    if visualize:
                        print("Gamma after gauge fixing", Gamma.reshape(Gamma.shape[0]**2, -1)[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
                        print("is_WTG:", is_WTG(Gamma, sigma))
                        print("cycle entropy:", calc_cycle_entropy(Gamma, sigma))
                        sig = np.diag(sigma)
                        sig = sig / np.linalg.norm(sig)
                        print("WTG coef:", sig)

                U, S, Vh, Fid = None, None, None, 1.0
                truncate_dim = None
                if threthold is not None:
                    for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
                        if cur_truncate_dim == Gamma.shape[0]:
                            print("no truncation done")
                            U = None
                            break
                        for sd in range(10):
                            U, S, Vh, Fid, trace = calc_optimal_truncation(Gamma, sigma, cur_truncate_dim, trials, visualize=visualize)
                            truncate_dim = cur_truncate_dim
                            if Fid > threthold:
                                break
                        if Fid > threthold:
                            break
                            
                # if truncation is executed        
                if U is not None:
                    if visualize:
                        Gamma = oe.contract("iIjJ,ip,qj,IP,QJ->pPqQ",Gamma,U,Vh,U.conj(),Vh.conj())
                        sigma = S
                        print("Gamma after optimal truncation", Gamma.reshape(Gamma.shape[0]**2, -1)[:max(5, Gamma.shape[0]),:max(5, Gamma.shape[1])])
                        print("Fid from Gamma, S:", oe.contract("iIjJ,ij,IJ",Gamma,sigma,sigma))
                        print("is_WTG:", is_WTG(Gamma, sigma))
                        print("cycle entropy:", calc_cycle_entropy(Gamma, sigma))
                        sig = np.diag(sigma)
                        sig = sig / np.linalg.norm(sig)
                        print("WTG coef:", sig)
                    U = np.dot(U, S) / np.sqrt(trace)
                    """news_dim = S.shape[0]
                    Tmp = np.dot(np.dot(U, S), Vh)
                    newU, news, newVh = np.linalg.svd(Tmp)
                    news = news[:news_dim]
                    newU = newU[:, :news_dim]
                    newVh = newVh[:news_dim, :]
                    U = np.dot(newU, np.diag(np.sqrt(news))) / np.sqrt(trace)
                    Vh = np.dot(np.diag(np.sqrt(news)), newVh)"""

                    trun_node_idx = h*self.width+w
                    trun_edge_idx = 2
                    op_node_idx = h*self.width+w+1
                    op_edge_idx = 4

                    if is_fix_gauge:
                        U = np.dot(xinv, U)
                        Vh = np.dot(Vh, yinv)

                    self.__apply_bond_matrix(trun_node_idx, trun_edge_idx, op_node_idx, op_edge_idx, U, Vh)

                    print(f"truncate dim: {truncate_dim}")
                    total_fid = total_fid * Fid
                    print(f"fidelity: {Fid}")
                    print(f"total fidelity: {total_fid}")

                    # also for mps_left_nodes, mps_right_list
                    Utensor = oe.contract("ij,IJ->jJiI",U,U.conj()).reshape(U.shape[1]**2,-1,1,1)
                    mps_left.apply_MPO([h], MPO([Utensor]))

                    Vhtensor = oe.contract("ij,IJ->iIjJ",Vh,Vh.conj()).reshape(Vh.shape[0]**2,-1,1,1)
                    mps_right_list[w+1].apply_MPO([self.height-1-h], MPO([Vhtensor]))

        return total_fid

    def calc_singularity(self):
        sing = 1.0
        for node in self.nodes:
            sing *= np.linalg.norm(node.tensor)
        return sing

    def regularize_PEPS(self, threshold=10, max_trial=20, visualize=False):
        sing = self.calc_singularity()
        if visualize:
            print(f"initial singularity: {sing}")
        trial = 0
        while sing > threshold and trial < max_trial:
            # vertical bond
            for h in range(self.height - 1):
                for w in range(self.width):
                    tensorup = self.nodes[h*self.width+w].tensor
                    shapeup = tensorup.shape
                    tensordown = self.nodes[(h+1)*self.width+w].tensor
                    shapedown = tensordown.shape
                    tmp = oe.contract("abcde,AdCDE->abceACDE", tensorup, tensordown).reshape(np.prod(shapeup[0:3]+shapeup[4:5]), -1)
                    U, s, Vh = np.linalg.svd(tmp, full_matrices=False)
                    sdim = min(shapeup[3], len(s))
                    s = s[:sdim]
                    U = np.dot(U[:,:sdim], np.diag(np.sqrt(s)))
                    Vh = np.dot(np.diag(np.sqrt(s)), Vh[:sdim,:])
                    self.nodes[h*self.width+w].tensor = U.reshape(shapeup[0],shapeup[1],shapeup[2],shapeup[4],sdim).transpose(0,1,2,4,3)
                    self.nodes[(h+1)*self.width+w].tensor = Vh.reshape(sdim,shapedown[0],shapedown[2],shapedown[3],shapedown[4]).transpose(1,0,2,3,4)

            # horizontal bond
            for w in range(self.width - 1):
                for h in range(self.height):
                    tensorleft = self.nodes[h*self.width+w].tensor
                    shapeleft = tensorleft.shape
                    tensorright = self.nodes[h*self.width+w+1].tensor
                    shaperight = tensorright.shape
                    tmp = oe.contract("abcde,ABCDc->abdeABCD", tensorleft, tensorright).reshape(-1, np.prod(shaperight[:4]))
                    U, s, Vh = np.linalg.svd(tmp, full_matrices=False)
                    sdim = min(shapeleft[2], len(s))
                    s = s[:sdim]
                    U = np.dot(U[:,:sdim], np.diag(np.sqrt(s)))
                    Vh = np.dot(np.diag(np.sqrt(s)), Vh[:sdim,:])
                    self.nodes[h*self.width+w].tensor = U.reshape(shapeleft[0],shapeleft[1],shapeleft[3],shapeleft[4],sdim).transpose(0,1,4,2,3)
                    self.nodes[h*self.width+w+1].tensor = Vh.reshape(sdim,shaperight[0],shaperight[1],shaperight[2],shaperight[3]).transpose(1,2,3,4,0)

            new_sing = self.calc_singularity()
            if visualize:
                print(f"trial {trial}, singularity: {new_sing}")
            if (sing - new_sing) / sing < 1e-5:
                if visualize:
                    print("no more improvement at PEPS regularization")
                    break
            trial += 1
            sing = new_sing
        return sing
    
    def prepare_Gamma_old(self, trun_node_idx):
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

        return trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order


    #def find_Gamma_tree(self, trun_node_idx, algorithm=None, memory_limit=None, visualize=False):
        """find contraction tree of Gamma

        Args:
            trun_node_idx (list ofint) : the node index connected to the target edge
            truncate_dim (int) : the target bond dimension
            trial (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        """for i in range(self.n):
            self.nodes[i].name = f"node{i}"

        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)

        tree, cost, sp_cost = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit, visualize=visualize)
        return tree, cost, sp_cost"""


    def find_optimal_truncation(self, trun_node_idx, min_truncate_dim=None, max_truncate_dim=None, truncate_buff=None, threthold=None, trials=None, gauge=False, algorithm=None, tnq=None, tree=None, target_size=None, gpu=True, thread=1, seq="ADCRS", visualize=False, calc_lim=None):
        """truncate the specified index using FET method

        Args:
            trun_node_idx (int) : the node index connected to the target edge
            truncate_dim (int) : the target bond dimension
            trial (int) : the number of iterations
            visualize (bool) : if printing the optimization process or not
        """
        for i in range(self.n):
            self.nodes[i].name = f"node{i}"

        trun_node_idx, op_node_idx, trun_edge_idx, op_edge_idx, node_list, output_edge_order = self.prepare_Gamma(trun_node_idx)

        if min_truncate_dim is not None and self.nodes[trun_node_idx][trun_edge_idx].dimension <= min_truncate_dim:
            print("trun_dim already satisfied")
            return 1.0

        max_truncate_dim = min(max_truncate_dim, self.nodes[trun_node_idx][trun_edge_idx].dimension)

        # includes tree == None case
        output_inds = None
        if tnq is None:
            tnq, output_inds = from_tn_to_quimb(node_list, output_edge_order)
            tnq, tree = self.find_contract_tree_by_quimb(tnq, output_inds, algorithm, seq, visualize)
            if calc_lim is not None and tree.total_flops() > calc_lim:
                print("Gamma calc lim exceeded.")
                return None
            #tnq, tree = self.find_Gamma_tree([trun_node_idx, op_node_idx], algorithm=algorithm, seq=seq, visualize=visualize)

        #print("calc Gamma...")
        Gamma = self.contract_tree_by_quimb(tn=tnq, tree=tree, output_inds=output_inds) #iIjJ


        print("Fid from Gamma, S:", oe.contract("iIiI",Gamma))

        U, Vh, Fid = None, None, 1.0
        truncate_dim = None
        if threthold is not None:
            for cur_truncate_dim in range(min_truncate_dim, max_truncate_dim+1, truncate_buff):
                if cur_truncate_dim == Gamma.shape[0]:
                    print("no truncation done")
                    return 1.0
                if not gauge:
                    U, Vh, Fid = self.find_optimal_truncation_by_Gamma(Gamma, cur_truncate_dim, trials, gpu=gpu, visualize=visualize)
                else:
                    U, Vh, Fid = self.fix_gauge_and_find_optimal_truncation_by_Gamma(Gamma, cur_truncate_dim, trials, gpu=gpu, visualize=visualize)
                
                truncate_dim = cur_truncate_dim
                if Fid > threthold:
                    break
        print(f"truncate dim: {truncate_dim}")

        """# truncate while Fid < threthold
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
            U, Vh, Fid = self.find_optimal_truncation_by_Gamma(Gamma, truncate_dim, trials, visualize=visualize)"""

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

    def calc_horizontal_BMPS(self, wstart, wend, hstart, hend, bmps_truncate_dim=None, bmps_threshold=1.0, visualize=False):
        """execute BMPS in horizontal direction and return mps

        Args:
            wstart, wend (int) : execute horizontal BMPS from [wstart, wend) or (wend, wstart]
            hstart, hend (int) : execute horizontal BMPS, supported on [hstart, hend) or (hend, hstart]
        """
        peps_tensors = []
        for idx in range(self.n):
            tmp = self.__contract_node_inner(idx)
            peps_tensors.append(tmp)

        total_fid = 1.0
        
        # horizontal BMPS
        mps_tensors = [np.array([1]).reshape(1,1,1) for _ in range(abs(hend - hstart))]
        mps_horizontal = MPS(mps_tensors, truncate_dim=bmps_truncate_dim, threthold_err=1-bmps_threshold)
        mps_horizontal.canonicalization()
        wstep = 1 if wstart < wend else -1
        hstep = 1 if hstart < hend else -1

        transpose_list = (1,3,0,2) if wstart < wend else (3,1,0,2)
        for w in range(wstart, wend, wstep):
            mpo_tensors = []
            for h in range(hstart, hend, hstep):
                tensor = peps_tensors[h*self.width+w]
                mpo_tensors.append(tensor.transpose(transpose_list))
            mpo = MPO(mpo_tensors)
            fid = mps_horizontal.apply_MPO([i for i in range(abs(hend - hstart))], mpo, is_normalize=False, last_dir=2)
            total_fid = total_fid * fid
            if visualize:
                print("bmps mps-dim for horizontal BMPS", mps_horizontal.virtual_dims)
        if visualize:
            print(f"total fidelity: {total_fid}")
        return mps_horizontal

    def calc_vertical_BMPS(self, hstart, hend, wstart, wend, bmps_truncate_dim=None, bmps_threshold=1.0, visualize=False):
        """execute BMPS in vertical direction and return mps

        Args:
            hstart, hend (int) : execute vertical BMPS from [hstart, hend) or (hend, hstart]
            wstart, wend (int) : execute vertical BMPS, supported on [wstart, wend) or (wend, wstart]
        """
        peps_tensors = []
        for idx in range(self.n):
            tmp = self.__contract_node_inner(idx)
            peps_tensors.append(tmp)

        total_fid = 1.0
        
        # vertical BMPS
        mps_tensors = [np.array([1]).reshape(1,1,1) for _ in range(abs(wend - wstart))]
        mps_vertical = MPS(mps_tensors, truncate_dim=bmps_truncate_dim, threthold_err=1-bmps_threshold)
        mps_vertical.canonicalization()
        hstep = 1 if hstart < hend else -1
        wstep = 1 if wstart < wend else -1

        transpose_list = (2,0,3,1) if hstart < hend else (0,2,3,1)
        for h in range(hstart, hend, hstep):
            mpo_tensors = []
            for w in range(wstart, wend, wstep):
                tensor = peps_tensors[h*self.width+w]
                mpo_tensors.append(tensor.transpose(transpose_list))
            mpo = MPO(mpo_tensors)
            fid = mps_vertical.apply_MPO([i for i in range(abs(wend - wstart))], mpo, is_normalize=False, last_dir=2)
            total_fid = total_fid * fid
            if visualize:
                print("bmps mps-dim for vertical BMPS", mps_vertical.virtual_dims)
        if visualize:
            print(f"total fidelity: {total_fid}")
        return mps_vertical

    def calc_full_environment(self, left_idx, right_idx, top_idx, down_idx, bmps_threshold=None, visualize=False):
        # return tensor which edges in order left, right, top, down

        mps_left = self.calc_horizontal_BMPS(0, left_idx, 0, self.height, bmps_threshold=bmps_threshold, visualize=visualize)
        mps_right = self.calc_horizontal_BMPS(self.width-1, right_idx, 0, self.height, bmps_threshold=bmps_threshold, visualize=visualize)
        mps_top = self.calc_vertical_BMPS(0, top_idx, left_idx, right_idx+1, bmps_threshold=bmps_threshold, visualize=visualize)
        mps_down = self.calc_vertical_BMPS(self.height-1, down_idx, left_idx, right_idx+1, bmps_threshold=bmps_threshold, visualize=visualize)

        def create_corner_tensor(mps, start, end):
            # contract [start, end) or (end, start] of mps

            # in case of out of range
            if start == end:
                return np.array([1]).reshape(1,1)

            ans = None
            step = 1 if start < end else -1
            for i in range(start, end, step):
                if ans is None:
                    ans = mps.tensors[i]
                else:
                    if step == 1:
                        lbdim = ans.shape[1]
                        rbdim = mps.tensors[i].shape[2]
                        ans = oe.contract("abc,dce->adbe",ans,mps.tensors[i]).reshape(-1, lbdim, rbdim)
                    else:
                        lbdim = mps.tnsors[i].shape[1]
                        rbdim = ans.shape[2]
                        ans = oe.contract("abc,deb->adec",ans,mps.tensors[i]).reshape(-1, lbdim, rbdim)
            return ans.reshape(ans.shape[0], -1)

        top_left = tn.Node(create_corner_tensor(mps_left, 0, top_idx))
        top_right = tn.Node(create_corner_tensor(mps_right, 0, top_idx))
        down_left = tn.Node(create_corner_tensor(mps_left, down_idx+1, self.height))
        down_right = tn.Node(create_corner_tensor(mps_right, down_idx+1, self.height))

        node_list = [top_left, top_right, down_left, down_right]
        output_edge_order = []

        # left
        for h in range(top_idx, down_idx+1):
            tmp = tn.Node(mps_left.tensors[h])
            output_edge_order.append(tmp[0])
            if h == top_idx:
                tn.connect(tmp[1], top_left[1])
            else:
                tn.connect(tmp[1], node_list[-1][2])
            if h == down_idx:
                tn.connect(tmp[2], down_left[1])
            node_list.append(tmp)  
        # right
        for h in range(top_idx, down_idx+1):
            tmp = tn.Node(mps_right.tensors[h])
            output_edge_order.append(tmp[0])
            if h == top_idx:
                tn.connect(tmp[1], top_right[1])
            else:
                tn.connect(tmp[1], node_list[-1][2])
            if h == down_idx:
                tn.connect(tmp[2], down_right[1])
            node_list.append(tmp)
        # top
        top_nodes = tn.replicate_nodes(mps_top.nodes)
        for w in range(len(top_nodes)):
            output_edge_order.append(top_nodes[w][0])
        tn.connect(top_nodes[0][1], top_left[0])
        tn.connect(top_nodes[-1][2], top_right[0])
        node_list += top_nodes
        # down
        down_nodes = tn.replicate_nodes(mps_down.nodes)
        for w in range(len(down_nodes)):
            output_edge_order.append(down_nodes[w][0])
        tn.connect(down_nodes[0][1], down_left[0])
        tn.connect(down_nodes[-1][2], down_right[0])
        node_list += down_nodes

        env = tn.contractors.auto(node_list, output_edge_order=output_edge_order).tensor
        env_shape = []
        env_transpose_list = [2*i for i in range(len(env.shape))] + [2*i+1 for i in range(len(env.shape))]
        env_dim = 1
        for s in env.shape:
            ss = int(np.sqrt(s))
            env_shape += [ss, ss]
            env_dim *= ss
        env = env.reshape(env_shape)
        env = env.transpose(env_transpose_list)
        env_eig = env.reshape(env_dim, env_dim)
        eig, w = np.linalg.eig(env_eig)
        print(eig)
        print(env.shape)
        return env

        """tensors = []
        tensors.append(create_corner_tensor(mps_left, 0, top_idx))
        tensors.append(mps_top.tensors[left_idx])
        tensors.append(mps_top.tensors[right_idx])
        tensors.append(create_corner_tensor(mps_right, 0, top_idx))
        tensors.append(mps_left.tensors[top_idx])
        tensors.append(mps_left.tensors[down_idx])
        tensors.append(mps_right.tensors[top_idx])
        tensors.append(mps_right.tensors[down_idx])
        tensors.append(create_corner_tensor(mps_left, down_idx+1, self.height))
        tensors.append(mps_down.tensors[left_idx])
        tensors.append(mps_down.tensors[right_idx])
        tensors.append(create_corner_tensor(mps_right, down_idx+1, self.height))

        env = oe.contract("ab,cab,edf,fg,hbi,jik,lgm,nmo,pk,qpr,srt,to->hjlnceqs",*tensors)
        env_shape = []
        env_dim = 1
        for s in env.shape:
            ss = int(np.sqrt(s))
            env_shape += [ss, ss]
            env_dim *= ss
        env = env.reshape(env_shape)
        env = env.transpose(0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15)
        env_eig = env.reshape(env_dim, env_dim)
        eig, w = np.linalg.eig(env_eig)
        print(eig)
        print(env.shape)
        return env"""

    def calc_simple_environment(self, left_idx, right_idx, top_idx, down_idx, visualize=False):
        mps_left = self.calc_horizontal_BMPS(0, left_idx, 0, self.height, 1, visualize=visualize)
        mps_right = self.calc_horizontal_BMPS(self.width-1, right_idx, 0, self.height, 1, visualize=visualize)
        mps_top = self.calc_vertical_BMPS(0, top_idx, 0, self.width, 1, visualize=visualize)
        mps_down = self.calc_vertical_BMPS(self.height-1, down_idx, 0, self.width, 1, visualize=visualize)

        def convert_simple_env(mps, start, end):
            return [tensor.reshape(int(np.sqrt(tensor.shape[0])), -1) for tensor in mps.tensors[start:end+1]]

        mps_left_tensors = convert_simple_env(mps_left, top_idx, down_idx)
        mps_right_tensors = convert_simple_env(mps_right, top_idx, down_idx)
        mps_top_tensors = convert_simple_env(mps_top, left_idx, right_idx)
        mps_down_tensors = convert_simple_env(mps_down, left_idx, right_idx)


        # return list of np.array in order left, right, top, down
        return mps_left_tensors, mps_right_tensors, mps_top_tensors, mps_down_tensors

    def calc_simple_ALS(self, left_idx, right_idx, top_idx, down_idx, als_truncate_dim, iters=10):
        """update tensor by ALS using simple environment

        Args:
            left_idx, right_idx, top_idx, down_idx (int) : the updating area [left, right], [top, down]
        """

        # initialize new tensor
        original_nodes = []
        for h in range(top_idx, down_idx+1):
            original_nodes += self.nodes[h*self.width+left_idx:h*self.width+right_idx+1]
        original_nodes = tn.replicate_nodes(original_nodes)
        new_nodes = tn.replicate_nodes(original_nodes)
        area_height = down_idx - top_idx + 1
        area_width = right_idx - left_idx + 1
        area_num = area_height * area_width

        for h in range(area_height):
            for w in range(area_width):
                shape = list(original_nodes[h*area_width+w].tensor.shape)
                if w != 0:
                    shape[4] = min(shape[4], als_truncate_dim)
                if w != area_width-1:
                    shape[2] = min(shape[2], als_truncate_dim)
                if h != 0:
                    shape[1] = min(shape[1], als_truncate_dim)
                if h != area_height-1:
                    shape[3] = min(shape[3], als_truncate_dim)
                new_nodes[h*area_width+w].tensor = np.random.randn(*shape)

        # define tensor A
        tensorA_nodes = tn.replicate_nodes(new_nodes) + tn.replicate_nodes(new_nodes)
        for idx in range(area_num, 2 * area_num):
            tensorA_nodes[idx].tensor = tensorA_nodes[idx].tensor.conj()
        for idx in range(area_num):
            tn.connect(tensorA_nodes[idx][0], tensorA_nodes[idx+area_num][0])

        # define tensor B
        tensorB_nodes = tn.replicate_nodes(original_nodes) + tn.replicate_nodes(new_nodes)
        for idx in range(area_num, 2 * area_num):
            tensorB_nodes[idx].tensor = tensorB_nodes[idx].tensor.conj()
        for idx in range(area_num):
            tn.connect(tensorB_nodes[idx][0], tensorB_nodes[idx+area_num][0])

        # connect simple env
        left_tensors, right_tensors, top_tensors, down_tensors = self.calc_simple_environment(left_idx, right_idx, top_idx, down_idx, visualize=False)
        
        def connect_simple_env(node_list):
            # top, down simple envs
            for w in range(area_width):
                top = tn.Node(top_tensors[w])
                tn.connect(top[0], node_list[w][1])
                tn.connect(top[1], node_list[w+area_num][1])
                node_list.append(top)
                down = tn.Node(down_tensors[w])
                tn.connect(down[0], node_list[(area_height-1)*area_width+w][3])
                tn.connect(down[1], node_list[(area_height-1)*area_width+w+area_num][3])
                node_list.append(down)

            # left, right simple envs
            for h in range(area_height):
                left = tn.Node(left_tensors[h])
                tn.connect(left[0], node_list[area_width*h][4])
                tn.connect(left[1], node_list[area_width*h+area_num][4])
                node_list.append(left)
                right = tn.Node(right_tensors[h])
                tn.connect(right[0], node_list[area_width*h+area_width-1][2])
                tn.connect(right[1], node_list[area_width*h+area_width-1+area_num][2])
                node_list.append(right)
            
            return node_list
        
        tensorA_nodes = connect_simple_env(tensorA_nodes)
        tensorB_nodes = connect_simple_env(tensorB_nodes)

        state_before = self.contract().flatten()
        state_before /= np.linalg.norm(state_before)

        original_tensors = self.tensors

        for h in range(area_height):
            for w in range(area_width):
                self.nodes[(h+top_idx)*self.width+(w+left_idx)].tensor = tensorA_nodes[h*area_width+w].tensor
                
        state_after = self.contract().flatten()
        state_after /= np.linalg.norm(state_after)
        past_fid = np.dot(state_before, state_after)
        print("initial fidelity:", past_fid)
        fidelity = 0.0

        for iter in range(iters):
            for h in range(area_height):
                for w in range(area_width):
                    # print(f"iter:{iter} h:{h} w:{h}")
                    tensor_shape = tensorA_nodes[h*area_width+w].tensor.shape
                    tensor_dim = np.prod(tensor_shape[1:])

                    contract_node_list = tn.replicate_nodes(tensorA_nodes)
                    contract_node_list[h*area_width+w][0].disconnect()
                    output_edge_orderA = contract_node_list[h*area_width+w+area_num].edges[1:]
                    output_edge_orderA += contract_node_list[h*area_width+w].edges[1:]
                    contract_node_list.pop(h*area_width+w+area_num)
                    contract_node_list.pop(h*area_width+w)

                    tensorA = tn.contractors.auto(contract_node_list, output_edge_order=output_edge_orderA).tensor.reshape(tensor_dim, -1)

                    contract_node_list = tn.replicate_nodes(tensorB_nodes)
                    output_edge_orderB = contract_node_list[h*area_width+w+area_num].edges
                    contract_node_list.pop(h*area_width+w+area_num)
                    tensorB = tn.contractors.auto(contract_node_list, output_edge_order=output_edge_orderB).tensor.reshape(-1, tensor_dim)

                    tensorAinv = np.linalg.pinv(tensorA)
                    newT = oe.contract("ab,cb->ca",tensorAinv,tensorB).reshape(tensor_shape)
                    tensorA_nodes[h*area_width+w].tensor = newT
                    tensorA_nodes[h*area_width+w+area_num].tensor = newT.conj()
                    tensorB_nodes[h*area_width+w+area_num].tensor = newT.conj()

            for h in range(area_height):
                for w in range(area_width):
                    self.nodes[(h+top_idx)*self.width+(w+left_idx)].tensor = tensorA_nodes[h*area_width+w].tensor

            state_after = self.contract().flatten()
            state_after /= np.linalg.norm(state_after)
            fidelity = np.dot(state_before, state_after)
            print("iter:", iter, "fidelity:", fidelity)
            if np.abs(fidelity - past_fid) < 1e-8:
                print("no more improvement")
                break
            past_fid = fidelity


        if fidelity < 1-1e-8:
            print("truncation failed")
            for i in range(self.height * self.width):
                self.nodes[i].tensor = original_tensors[i]
            return None
        return fidelity

    def calc_full_ALS(self, left_idx, right_idx, top_idx, down_idx, als_truncate_dim, threshold=1e-8, bmps_threshold=None, iters=10):
        """update tensor by ALS using simple environment

        Args:
            left_idx, right_idx, top_idx, down_idx (int) : the updating area [left, right], [top, down]
        """

        # initialize new tensor
        original_nodes = []
        for h in range(top_idx, down_idx+1):
            original_nodes += self.nodes[h*self.width+left_idx:h*self.width+right_idx+1]
        original_nodes = tn.replicate_nodes(original_nodes)
        new_nodes = tn.replicate_nodes(original_nodes)
        area_height = down_idx - top_idx + 1
        area_width = right_idx - left_idx + 1
        area_num = area_height * area_width

        for h in range(area_height):
            for w in range(area_width):
                shape = list(original_nodes[h*area_width+w].tensor.shape)
                if w != 0:
                    shape[4] = min(shape[4], als_truncate_dim)
                if w != area_width-1:
                    shape[2] = min(shape[2], als_truncate_dim)
                if h != 0:
                    shape[1] = min(shape[1], als_truncate_dim)
                if h != area_height-1:
                    shape[3] = min(shape[3], als_truncate_dim)
                new_nodes[h*area_width+w].tensor = np.random.randn(*shape)

        # define tensor A
        tensorA_nodes = tn.replicate_nodes(new_nodes) + tn.replicate_nodes(new_nodes)
        for idx in range(area_num, 2 * area_num):
            tensorA_nodes[idx].tensor = tensorA_nodes[idx].tensor.conj()
        for idx in range(area_num):
            tn.connect(tensorA_nodes[idx][0], tensorA_nodes[idx+area_num][0])

        # define tensor B
        tensorB_nodes = tn.replicate_nodes(original_nodes) + tn.replicate_nodes(new_nodes)
        for idx in range(area_num, 2 * area_num):
            tensorB_nodes[idx].tensor = tensorB_nodes[idx].tensor.conj()
        for idx in range(area_num):
            tn.connect(tensorB_nodes[idx][0], tensorB_nodes[idx+area_num][0])

        # connect full env
        # left1, left2, right1, right2, top1, top2, down1, down2, (its conj ...)
        env_tensors = self.calc_full_environment(left_idx, right_idx, top_idx, down_idx, bmps_threshold, visualize=False)
        
        def connect_full_env(node_list):
            env = tn.Node(env_tensors)
            edge_num = len(env_tensors.shape) // 2

            # left, right, top, down
            for h in range(area_height):
                tn.connect(env[0+h], node_list[area_width*h][4])
                tn.connect(env[edge_num+h], node_list[area_width*h+area_num][4])
            for h in range(area_height):
                tn.connect(env[area_height+h], node_list[area_width*h+area_width-1][2])
                tn.connect(env[area_height+edge_num+h], node_list[area_width*h+area_width-1+area_num][2])

            for w in range(area_width):
                tn.connect(env[2*area_height+w], node_list[w][1])
                tn.connect(env[2*area_height+edge_num+w], node_list[w+area_num][1])
                tn.connect(env[2*area_height+area_width+w], node_list[(area_height-1)*area_width+w][3])
                tn.connect(env[2*area_height+area_width+edge_num+w], node_list[(area_height-1)*area_width+w+area_num][3])
            
            node_list.append(env)
            return node_list
        
        tensorA_nodes = connect_full_env(tensorA_nodes)
        tensorB_nodes = connect_full_env(tensorB_nodes)

        state_before = self.contract().flatten()
        state_before /= np.linalg.norm(state_before)

        original_tensors = self.tensors

        for h in range(area_height):
            for w in range(area_width):
                self.nodes[(h+top_idx)*self.width+(w+left_idx)].tensor = tensorA_nodes[h*area_width+w].tensor
                
        state_after = self.contract().flatten()
        state_after /= np.linalg.norm(state_after)
        past_fid = np.dot(state_before, state_after)
        print("initial fidelity:", past_fid)
        fidelity = 0.0

        for iter in range(iters):
            for h in range(area_height):
                for w in range(area_width):
                    # print(f"iter:{iter} h:{h} w:{h}")
                    tensor_shape = tensorA_nodes[h*area_width+w].tensor.shape
                    tensor_dim = np.prod(tensor_shape[1:])

                    contract_node_list = tn.replicate_nodes(tensorA_nodes)
                    contract_node_list[h*area_width+w][0].disconnect()
                    output_edge_orderA = contract_node_list[h*area_width+w+area_num].edges[1:]
                    output_edge_orderA += contract_node_list[h*area_width+w].edges[1:]
                    contract_node_list.pop(h*area_width+w+area_num)
                    contract_node_list.pop(h*area_width+w)

                    tensorA = tn.contractors.auto(contract_node_list, output_edge_order=output_edge_orderA).tensor.reshape(tensor_dim, -1)

                    contract_node_list = tn.replicate_nodes(tensorB_nodes)
                    output_edge_orderB = contract_node_list[h*area_width+w+area_num].edges
                    contract_node_list.pop(h*area_width+w+area_num)
                    tensorB = tn.contractors.auto(contract_node_list, output_edge_order=output_edge_orderB).tensor.reshape(-1, tensor_dim)

                    tensorAinv = np.linalg.pinv(tensorA)
                    newT = oe.contract("ab,cb->ca",tensorAinv,tensorB).reshape(tensor_shape)
                    tensorA_nodes[h*area_width+w].tensor = newT
                    tensorA_nodes[h*area_width+w+area_num].tensor = newT.conj()
                    tensorB_nodes[h*area_width+w+area_num].tensor = newT.conj()

            for h in range(area_height):
                for w in range(area_width):
                    self.nodes[(h+top_idx)*self.width+(w+left_idx)].tensor = tensorA_nodes[h*area_width+w].tensor

            if iter == iters - 1:
                state_after = self.contract().flatten()
                state_after /= np.linalg.norm(state_after)
                fidelity = np.dot(state_before, state_after)
                print("iter:", iter, "fidelity:", fidelity)
                #if np.abs(fidelity - past_fid) < 1e-8:
                #    print("no more improvement")
                #    break
                #past_fid = fidelity
                break


        if fidelity < threshold:
            print("truncation failed")
            for i in range(self.height * self.width):
                self.nodes[i].tensor = original_tensors[i]
            return None
        return fidelity