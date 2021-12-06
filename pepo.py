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

class PEPO(TensorNetwork):
    """class of PEPDO

    physical bond: 0, 1, ..., n-1
    conj physical bond: n, n+1, ..., 2n-1
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
                edge_info.append([i, i+self.n, 2*self.n+w*(self.height+1)+h, buff+h*(self.width+1)+w+1, 2*self.n+w*(self.height+1)+h+1, buff+h*(self.width+1)+w])
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


    @property
    def inner_dims(self):
        inner_dims = []
        for i in range(self.n):
            inner_dims.append(self.nodes[i].get_dimension(5))
        return inner_dims


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

    
    def calc_trace(self, algorithm=None, memory_limit=None, path=None, visualize=False):
        """contract all PEPDO and generate trace of full density operator
        
        Returns:
            np.array: tensor after contraction
        """
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

        """for i in range(2*self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                print(i, dangling)
                output_edge_order.append(dangling)"""

        if path == None:
            path, total_cost = self.calc_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        self.path = path
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor


    def calc_pepo_trace(self, pepo, algorithm=None, memory_limit=None, path=None, visualize=False):
        """contract all PEPDO and generate trace of full density operator
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(pepo.nodes))
        for i in range(self.n):
            if cp_nodes[i].get_dimension(0) != 1:
                tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
            if cp_nodes[i].get_dimension(1) != 1:
                tn.connect(cp_nodes[i][1], cp_nodes[i+self.n][1])

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)
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

        # 5,4,3,2,1,0の順に消す
        # 表，裏
        for i in range(2):
            for h in range(self.height):
                if cp_nodes[i*self.n+h*self.width].get_dimension(5) == 1:
                    clear_dangling(i*self.n+h*self.width, 5)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+h*self.width][5])
            for w in range(self.width):
                if cp_nodes[i*self.n+self.width*(self.height-1)+w].get_dimension(4) == 1:
                    clear_dangling(i*self.n+self.width*(self.height-1)+w, 4)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+self.width*(self.height-1)+w][4])
            for h in range(self.height):
                if cp_nodes[i*self.n+(h+1)*self.width-1].get_dimension(3) == 1:
                    clear_dangling(i*self.n+(h+1)*self.width-1, 3)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+(h+1)*self.width-1][3])
            for w in range(self.width):
                if cp_nodes[i*self.n+w].get_dimension(2) == 1:
                    clear_dangling(i*self.n+w, 2)
                else:
                    output_edge_order.append(cp_nodes[i*self.n+w][2])

        # conj-physical, physical
        for i in range(self.n):
            if cp_nodes[i].get_dimension(1) == 1:
                clear_dangling(i, 1)
                clear_dangling(i+self.n, 1)
        
        for i in range(self.n):
            if cp_nodes[i].get_dimension(0) == 1:
                clear_dangling(i, 0)
                clear_dangling(i+self.n, 0)

        return cp_nodes, output_edge_order