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
            w_virtual_dims = [self.nodes[w].get_dimension(2)]
            for h in range(self.height):
                w_virtual_dims.append(self.nodes[w+h*self.width].get_dimension(4))
            virtual_dims.append(w_virtual_dims)
        return virtual_dims


    @property
    def horizontal_virtual_dims(self):
        virtual_dims = []
        for h in range(self.height):
            h_virtual_dims = [self.nodes[self.width].get_dimension(5)]
            for w in range(self.width):
                h_virtual_dims.append(self.nodes[w+h*self.width].get_dimension(3))
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

    
    def prepare_pepo_trace(self, pepo):
        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(pepo.nodes))
        for i in range(self.n):
            if cp_nodes[i].get_dimension(0) != 1:
                tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
            if cp_nodes[i].get_dimension(1) != 1:
                tn.connect(cp_nodes[i][1], cp_nodes[i+self.n][1])

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes1, output_edge_order1 = self.__clear_dangling(cp_nodes[:self.n])
        cp_nodes2, output_edge_order2 = self.__clear_dangling(cp_nodes[self.n:])
        cp_nodes = cp_nodes1 + cp_nodes2
        output_edge_order = output_edge_order1 + output_edge_order2
        node_list = [node for node in cp_nodes]

        return node_list, output_edge_order

    
    def find_pepo_trace_tree(self, pepo, algorithm=None, memory_limit=None, visualize=False):
        
        node_list, output_edge_order = self.prepare_pepo_trace(pepo)
        tree, total_cost, max_sp_cost = self.find_contract_tree(node_list, output_edge_order, algorithm, memory_limit, visualize=visualize)
        self.pepo_trace_tree = tree
        return tree, total_cost, max_sp_cost


    def calc_pepo_trace(self, pepo, algorithm=None, memory_limit=None, tree=None, path=None, visualize=False):
        """ calc inner-product and generate trace of full density operator
        
        Returns:
            np.array: tensor after contraction
        """

        node_list, output_edge_order = self.prepare_pepo_trace(pepo)
        #if path == None:
        #    path, total_cost = self.find_contract_path(node_list, algorithm=algorithm, memory_limit=memory_limit, output_edge_order=output_edge_order, visualize=visualize)
        #self.path = path
        if tree == None and path == None and self.pepo_trace_tree is not None:
            tree = self.pepo_trace_tree
        
        #return tn.contractors.contract_path(path, node_list, output_edge_order).tensor
        return self.contract_tree(node_list, output_edge_order, algorithm, memory_limit, tree, path, visualize=visualize)
    

    def find_optimal_truncation(self, trun_node_idx, trun_edge_idx, truncate_dim, algorithm=None, memory_limit=None):
        # not supported for trun_edge_idx=4or5 (do to the order of edge_disconnecting)
        op_node_idx = 0
        op_edge_idx = 0
        if trun_edge_idx == 2:
            op_node_idx = trun_node_idx
            trun_node_idx -= self.width
            op_edge_idx = 2
            trun_edge_idx = 4
        elif trun_edge_idx == 3:
            op_node_idx += 1
            op_edge_idx = 5
        elif trun_edge_idx == 4:
            op_node_idx = trun_node_idx + self.width
            op_edge_idx = 2
        else:
            op_node_idx = trun_node_idx
            trun_node_idx -= 1
            op_edge_idx = 5
            trun_edge_idx = 3

        print(trun_node_idx, trun_edge_idx)

        cp_nodes = tn.replicate_nodes(self.nodes)
        cp_nodes.extend(tn.replicate_nodes(self.nodes))
        for i in range(self.n):
            cp_nodes[i+self.n].tensor = cp_nodes[i+self.n].tensor.conj()
            if cp_nodes[i].get_dimension(0) != 1:
                tn.connect(cp_nodes[i][0], cp_nodes[i+self.n][0])
            if cp_nodes[i].get_dimension(1) != 1:
                tn.connect(cp_nodes[i][1], cp_nodes[i+self.n][1])
        
        edge_i, edge_j = cp_nodes[trun_node_idx][trun_edge_idx].disconnect("i", "j")
        edge_I, edge_J = cp_nodes[trun_node_idx+self.n][trun_edge_idx].disconnect("I", "J")
        output_edge_order = [edge_i, edge_I, edge_j, edge_J]

        # if there are dangling edges which dimension is 1, contract first (including inner dim)
        cp_nodes1, output_edge_order1 = self.__clear_dangling(cp_nodes[:self.n])
        cp_nodes2, output_edge_order2 = self.__clear_dangling(cp_nodes[self.n:])
        cp_nodes = cp_nodes1 + cp_nodes2
        output_edge_order.extend(output_edge_order1 + output_edge_order2)

        node_list = [node for node in cp_nodes]

        Gamma = self.contract_tree(node_list, output_edge_order, algorithm, memory_limit)

        bond_dim = Gamma.shape[0]
        trun_dim = truncate_dim
        print(f"bond: {bond_dim}, trun: {trun_dim}")

        I = np.eye(bond_dim)
        U, s, Vh = np.linalg.svd(I)
        U = U[:,:trun_dim]
        S = np.diag(s[:trun_dim])
        Vh = Vh[:trun_dim, :]
        print(U.shape, S.shape, Vh.shape)

        Fid = oe.contract("iIiI", Gamma)
        print(f"Fid before truncation: {Fid}")

        R = oe.contract("pq,qj->pj",S,Vh).flatten()
        P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
        A = oe.contract("a,b->ab",P,P.conj())
        B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)
        Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))
        print(f"Fid before optimization: {Fid}")

        for i in range(10):
            ## step1
            R = oe.contract("pq,qj->pj",S,Vh).flatten()
            P = oe.contract("iIjJ,ij,IP->PJ",Gamma,I,U.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,ip,IP->PJpj",Gamma,U,U.conj()).reshape(trun_dim*bond_dim, -1)

            #Fid = np.dot(R.conj(), np.dot(A, R)) / np.dot(R.conj(), np.dot(B, R))

            Rmax = np.dot(np.linalg.inv(B), P)
            Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
            print(f"fid at trial {i} step1: {Fid}")

            Utmp, stmp, Vh = np.linalg.svd(Rmax.reshape(trun_dim, -1), full_matrices=False)
            S = np.dot(Utmp, np.diag(stmp))

            """Binv = np.linalg.inv(B)
            Aprime = np.dot(Binv, A)
            eig, w = np.linalg.eig(Aprime)
            print(eig)"""

            ## step2
            R = oe.contract("ip,pq->qi",U,S).flatten()
            P = oe.contract("iIjJ,ij,QJ->QI",Gamma,I,Vh.conj()).flatten()
            A = oe.contract("a,b->ab",P,P.conj())
            B = oe.contract("iIjJ,qj,QJ->QIqi",Gamma,Vh,Vh.conj()).reshape(trun_dim*bond_dim, -1)

            Rmax = np.dot(np.linalg.inv(B), P)
            Fid = np.dot(Rmax.conj(), np.dot(A, Rmax)) / np.dot(Rmax.conj(), np.dot(B, Rmax))
            print(f"fid at trial {i} step2: {Fid}")

            U, stmp, Vhtmp = np.linalg.svd(Rmax.reshape(trun_dim, -1).T, full_matrices=False)
            S = np.dot(np.diag(stmp), Vhtmp)
        
        U = np.dot(U, S)
        Unode = tn.Node(U)
        Vhnode = tn.Node(Vh)
        tn.connect(Unode[1], Vhnode[0])

        left_edge, right_edge = self.nodes[trun_node_idx][trun_edge_idx].disconnect()
        edge_nodes = right_edge.get_nodes()
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