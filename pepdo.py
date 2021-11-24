from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
from mpo import MPO
from mps import MPS
from general_tn import TensorNetwork

class PEPDO(TensorNetwork):
    """class of PEPDO

    physical bond: 0, 1, ..., n-1
    inner bond: n, n+1, ..., 2n-1
    vertical virtual bond: 2n, 2n+1, ..., 2n+(height+1)-1, 2n+(height1), ..., 2n+(height+1)*width-1
    horizontal virtual bond: 2n+(height+1)*width, ..., 2n+(height+1)*width+(width+1)-1, ..., 2n+(height+1)*height+(height+1)*width-1

    Attributes:
        width (int) : PEPDO width
        height (int) : PEPDO height
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


    def contract(self, algorithm=None, memory_limit=None, path=None, visualize=False):
        """contract PEPDO and generate full density operator

        Args:
            output_edge_order (list of tn.Edge) : the order of output edge
        
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
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)

        if path == None:
            path, total_cost = self.calc_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        self.path = path
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor

    
    def amplitude_BMPS(self, tensors):
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        # contract product state first
        for i in range(self.n):
            state = tn.Node(tensors[i])
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
            print("bmps mps-dim", mps.virtual_dims)
            total_fid = total_fid * fid
            
        return mps.contract().flatten()[0]

    
    def amplitude(self, tensors, algorithm=None, memory_limit=None, path=None, visualize=False):
        """contract amplitude with given product states (typically computational basis)

        Args:
            tensor (list of np.array) : the given index represented by the list of tensor
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        # contract product state first
        for i in range(self.n):
            state = tn.Node(tensors[i])
            tn.connect(cp_nodes[i][0], state[0])
            edge_order = [cp_nodes[i].edges[j] for j in range(1, len(cp_nodes[i].edges))]
            cp_nodes[i] = tn.contractors.auto([cp_nodes[i], state], edge_order)
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)

        node_list = [node for node in cp_nodes]
        
        if path == None:
            path, total_cost = self.calc_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        self.path = path
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor
    
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
                    dir = return_dir(tidx[i] - tidx[i-1])
                    l_l_edges = [node_list[i-1][j] for j in range(0, 6) if j != dir]
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
                    node_list.append(new_node)

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]