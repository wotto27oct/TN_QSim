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
        self.tree, self.trace_tree, self.pepo_trace_tree = None, None, None


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
        """contract PEPO and generate full density operator

        Args:
            output_edge_order (list of tn.Edge) : the order of output edge
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][1])
            
        node_list = [node for node in cp_nodes]

        """for i in range(2*self.n):
            for dangling in cp_nodes[i].get_all_dangling():
                output_edge_order.append(dangling)

        if path == None:
            path, total_cost, _ = self.find_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        self.path = path
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor"""

        #if tree == None and path == None:
        #    tree, _, _  = self.find_contract_tree(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        #    self.tree = tree

        if tree == None and path == None and self.tree is not None:
            tree = self.tree

        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)
        
        #return tn.contractors.contract_path(path, node_list, output_edge_order).tensor

    
    def prepare_trace(self):
        cp_nodes = tn.replicate_nodes(self.nodes)

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes, output_edge_order = self.__clear_dangling(cp_nodes)

        node_list = []
        for i in range(self.n):
            tn.connect(cp_nodes[i][0], cp_nodes[i][1])
            node = tn.network_operations.contract_trace_edges(cp_nodes[i])
            node_list.append(node)

        return node_list, output_edge_order

    
    def calc_trace(self, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """contract all PEPO and generate trace of full density operator
        
        Returns:
            np.array: tensor after contraction
        """

        node_list, output_edge_order = self.prepare_trace()

        """if path == None:
            path, total_cost, _ = self.find_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        self.path = path
        return tn.contractors.contract_path(path, node_list, output_edge_order).tensor"""
        
        if tree == None and path == None and self.trace_tree is not None:
            tree = self.trace_tree

        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)    


    def find_trace_path(self, algorithm=None, memory_limit=None, visualize=False):
        """find contraction path of trace
        
        Returns:
            tree (ctg.ContractionTree) : the contraction tree
            total_cost (int) : total temporal cost
            max_sp_cost (int) : max spatial cost
        """
        node_list, output_edge_order = self.prepare_trace()

        tree, total_cost, max_sp_cost = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit, visualize=visualize)
        self.trace_tree = tree
        return tree, total_cost, max_sp_cost


    def calc_pepo_trace(self, pepo, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """ calc inner-product and generate trace of full density operator
        
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

        #if path == None:
        #    path, total_cost = self.find_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        #self.path = path
        if tree == None and path == None and self.pepo_trace_tree is not None:
            tree = self.pepo_trace_tree
        
        #return tn.contractors.contract_path(path, node_list, output_edge_order).tensor
        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)


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
        for h in range(self.height):
            if cp_nodes[h*self.width].get_dimension(5) == 1:
                clear_dangling(h*self.width, 5)
            else:
                output_edge_order.append(cp_nodes[h*self.width][5])
        for w in range(self.width):
            if cp_nodes[self.width*(self.height-1)+w].get_dimension(4) == 1:
                clear_dangling(self.width*(self.height-1)+w, 4)
            else:
                output_edge_order.append(cp_nodes[self.width*(self.height-1)+w][4])
        for h in range(self.height):
            if cp_nodes[(h+1)*self.width-1].get_dimension(3) == 1:
                clear_dangling((h+1)*self.width-1, 3)
            else:
                output_edge_order.append(cp_nodes[(h+1)*self.width-1][3])
        for w in range(self.width):
            if cp_nodes[w].get_dimension(2) == 1:
                clear_dangling(w, 2)
            else:
                output_edge_order.append(cp_nodes[w][2])

        # conj-physical, physical
        for i in range(self.n):
            if cp_nodes[i].get_dimension(1) == 1:
                clear_dangling(i, 1)
        
        for i in range(self.n):
            if cp_nodes[i].get_dimension(0) == 1:
                clear_dangling(i, 0)

        return cp_nodes, output_edge_order