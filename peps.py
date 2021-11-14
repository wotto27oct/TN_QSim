from math import trunc
from typing import ValuesView
import numpy as np
from numpy.core.fromnumeric import _reshape_dispatcher
import opt_einsum as oe
import cotengra as ctg
import tensornetwork as tn
from general_tn import TensorNetwork

class PEPS(TensorNetwork):
    """class of PEPS

    physical bond: 0, 1, ..., n-1
    vertical virtual bond: n, n+1, ..., n+(height+1)-1, n+(height1), ..., n+(height+1)*width-1
    horizontal virtual bond: n+(height+1)*width, ..., n+(height+1)*width+(width+1)-1, ..., n + (height+1)*height + (height+1)*width-1

    Attributes:
        width (int) : PEPS width
        height (int) : PEPS height
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
        truncate_dim (int) : truncation dim of virtual bond, default None
        threthold_err (float) : the err threthold of singular values we keep
    """

    def __init__(self, tensors, height, width, truncate_dim=None, threthold_err=None):
        self.n = len(tensors)
        self.height = height
        self.width = width
        edge_info = []
        buff = self.n + (self.height+1)*self.width
        for h in range(self.height):
            for w in range(self.width):
                i = h*self.width + w
                edge_info.append([i, self.n+w*(self.height+1)+h, buff+h*(self.width+1)+w+1, self.n+w*(self.height+1)+h+1, buff+h*(self.width+1)+w])
        super().__init__(edge_info, tensors)
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err


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


    def contract(self, algorithm=None, memory_limit=None, visualize=False):
        """contract whole PEPS and generate full state (+alpha)

        Args:
            output_edge_order (list of tn.Edge) : the order of output edge
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = [node for node in cp_nodes]

        for i in range(self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)

        path, total_cost = self.calc_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor
        #return tn.contractors.auto(node_list, output_edge_order=output_edge_order).tensor

    
    def amplitude(self, tensors, algorithm=None, memory_limit=None, visualize=False):
        """contract amplitude with given product states (typically computational basis)

        Args:
            tensor (list of np.array) : the given index represented by the list of tensor
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        for i in range(self.n):
            state = tn.Node(tensors[i])
            tn.connect(cp_nodes[i][0], state[0])
            edge_order = [cp_nodes[i].edges[j] for j in range(1, len(cp_nodes[i].edges))]
            cp_nodes[i] = tn.contractors.auto([cp_nodes[i], state], edge_order)
            #node_list.append(state)
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)

        node_list = [node for node in cp_nodes]
        
        #opt = ctg.HyperOptimizer(methods="spinglass")
        #opt = oe.paths.greedy()
        #return tn.contractors.custom(node_list, output_edge_order=output_edge_order, optimizer=opt).tensor
        path, total_cost = self.calc_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor
        #return tn.contractors.optimal(node_list, output_edge_order=output_edge_order).tensor

    
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

    def test_contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)
        output_edge_order = []
        # if there are dangling edges which dimension is 1, contract together
        def clear_dangling(node_idx, dangling_index):
            one = tn.Node(np.array([1]))
            edge = tn.connect(cp_nodes[node_idx][dangling_index], one[0])
            output_edge_order = []
            for i in range(len(cp_nodes[node_idx].edges)):
                if i != dangling_index:
                    output_edge_order.append(cp_nodes[node_idx][i])
            cp_nodes[node_idx] = tn.contractors.auto([cp_nodes[node_idx], one], output_edge_order)
            #node_list.append(one)

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

        for i in range(self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)

        node_list = [node for node in cp_nodes]
        self.contract_by_oe(node_list)
        return None


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
            edge_list.append(node_edge_list)
        
        for i in range(len(tidx)):
            if i != len(tidx)-1:
                dir = return_dir(tidx[i+1] - tidx[i]) 
                edge_list[i][dir] = tn.flatten_edges([edge_list[i][dir], edge_list[i][-1]])
                edge_list[i+1][(dir+1)%4+1] = edge_list[i][dir]
            if i != 0 or i != len(tidx)-1:
                edge_list[i].pop()
            self.nodes[tidx[i]] = node_list[i].reorder_edges(edge_list[i])