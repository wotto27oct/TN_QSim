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

    def find_contract(self,algorithm=None, seq="ADCRS", visualize=False):
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
            tensors (list of np.array) : the amplitude index represented by the list of tensor
            algorithm : the algorithm to find contraction path

        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_contract()
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)