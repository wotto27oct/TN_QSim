import numpy as np
from scipy.sparse.linalg.eigen.arpack.arpack import CNEUPD_ERRORS
import tensornetwork as tn
from tn_qsim.mpo import MPO
from tn_qsim.mps import MPS
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.utils import from_tn_to_quimb

class MERA(TensorNetwork):
    """class of binary MERA

    bond index: (0,1), (2,3,4,5), (6,7,8,9), (10,11,12,13,14,15,16,17), ...

    edge index order
    isometry: top, left, right
    unitary: topleft, topright, bottomleft, bottomright

    Attributes:
        depth (int) : the depth of MERA, >=1
        n (int) : the number of tensors
        edges (list of tn.Edge) : the list of each edge connected to each tensor
        nodes (list of tn.Node) : the list of each tensor
    """

    def __init__(self, tensors, depth):
        self.n = len(tensors)
        self.depth = depth
        edge_info = []
        buff = 0
        for d in range(self.depth):
            if d == 0:
                edge_info.append([0, 1])
                continue
            # depth=d isometry
            buff2 = buff + 2**d
            for w in range(2**d):
                edge_info.append([buff+w, buff2+2*w, buff2+2*w+1])
            buff = buff2 + 2**(d+1)
            for w in range(2**d):
                edge_info.append([buff2+2*w+1,buff2+(2*w+2)%(2**(d+1)),buff+2*w+1,buff+(2*w+2)%(2**(d+1))])
        super().__init__(edge_info, tensors)

    
    def prepare_contract(self):
        cp_nodes = tn.replicate_nodes(self.nodes)

        output_edge_order = []
        if self.depth == 1:
            output_edge_order = [cp_nodes[0][0], cp_nodes[0][1]]
        else:
            output_edge_order.append(cp_nodes[-1][3])
            for q in reversed(list(range(2**(self.depth-1)-1))):
                output_edge_order.append(cp_nodes[-(q+2)][2])
                output_edge_order.append(cp_nodes[-(q+2)][3])
            output_edge_order.append(cp_nodes[-1][2])

        node_list = [node for node in cp_nodes]

        return node_list, output_edge_order

    def find_contract(self, algorithm=None, seq="ADCRS", visualize=False):
        """contract contraction path by using quimb

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """
        
        node_list, output_edge_order = self.prepare_contract()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def contract(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """contract MERA and generate full state

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_contract()
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)


    def prepare_inner(self):
        node_list1, output_edge_order1 = self.prepare_contract()
        node_list2, output_edge_order2 = self.prepare_contract()
        for node in node_list2:
            node.tensor = node.tensor.conj()

        for q in range(2**self.depth):
            tn.connect(output_edge_order1[q], output_edge_order2[q])
        
        node_list = node_list1 + node_list2
        return node_list, []


    def find_calc_inner(self, algorithm=None, seq="ADCRS", visualize=False):
        """find calc_inner contraction path by using quimb

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """
        
        node_list, output_edge_order = self.prepare_inner()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def calc_inner(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """calc inner product of MERA state

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_inner()
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)


    def prepare_second_renyi(self, Arange):
        node_list1, output_edge_order1 = self.prepare_contract()
        node_list2, output_edge_order2 = self.prepare_contract()
        node_list3, output_edge_order3 = self.prepare_contract()
        node_list4, output_edge_order4 = self.prepare_contract()

        for node in node_list1:
            node.tensor = node.tensor.conj()
        for node in node_list3:
            node.tensor = node.tensor.conj()

        # connect open indices
        for a in Arange:
            tn.connect(output_edge_order2[a], output_edge_order3[a])
            tn.connect(output_edge_order4[a], output_edge_order1[a])
        for b in range(2**self.depth):
            if b not in Arange:
                tn.connect(output_edge_order1[b], output_edge_order2[b])
                tn.connect(output_edge_order3[b], output_edge_order4[b])

        output_edge_order = []

        node_list = node_list1 + node_list2 + node_list3 + node_list4

        return node_list, output_edge_order

    def find_calc_second_renyi(self, Arange, algorithm=None, seq="ADCRS", visualize=False):
        """contract contraction path by using quimb

        Args:
            Arange (list of int) : the range of A
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """
        
        node_list, output_edge_order = self.prepare_second_renyi(Arange)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def calc_second_renyi(self, Arange=None, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """contract contraction path by using quimb

        Args:
            Arange (int) : the range of A, [0, Arange-1]
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """
        output_inds = None
        if tn is None or tree is None:
            node_list, output_edge_order = self.prepare_second_renyi(Arange)
            tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, output_inds, target_size, gpu, thread, seq)

    
    def prepare_foliation(self, cut_list):
        node_list, output_edge_order = self.prepare_inner()

        rho_edge_order1 = []
        rho_edge_order2 = []
        for node_idx1, edge_idx1, node_idx2, edge_idx2 in cut_list:
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

    
    def prepare_grad_foliation(self, tidx, output_list):
        node_list1, output_edge_order1 = self.prepare_contract()
        node_list2, output_edge_order2 = self.prepare_contract()
        for node in node_list2:
            node.tensor = node.tensor.conj()
        for q in range(2**self.depth):
            tn.connect(output_edge_order1[q], output_edge_order2[q])

        node_list = []
        output_edge_order = []
        for nidx, eidx in output_list:
            output_edge_order.append(node_list1[nidx][eidx])
        for t in tidx:
            node_list.append(node_list1[t])
        node_list += node_list2

        return node_list, output_edge_order


    def find_calc_grad_foliation(self, tidx, output_list, algorithm=None, seq="ADCRS", visualize=False):
        """find calc_foliation contraction path by using quimb

        Args:
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_grad_foliation(tidx, output_list)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def calc_grad_foliation(self, tidx=None, output_list=None, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """calc foliation of MERA state

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_grad_foliation(tidx, output_list)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)