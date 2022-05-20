import numpy as np
from numpy.core.fromnumeric import reshape
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.mpo import MPO
from tn_qsim.utils import from_tn_to_quimb
import jax
import functools

class MERA3D(TensorNetwork):
    """class of MERA3D for renormalization

    physical bond: 0, 1, ..., n-1

    Args:
        n (int) : the number of qubits
        H (np.array) : Hamiltonian which we renormalize
        tidx (list of int) : the support of Hamiltonian

    Attributes:
        n (int) : the number of qubits
        Hnode (tn.Node) : Hamiltonian which we renormalize
        top_edges (list of (int, int)) : the list of top edges, (node_idx, edge_idx of node)
        top_nodes (list of tn.Node) : the list of nodes
        down_nodes (list of tn.Node) : the list of adjoint nodes
    """

    def __init__(self, n, H, tidx):
        self.n = n
        self.top_edges = [None for _ in range(n)]
        self.down_edges = [None for _ in range(n)]
        self.Hnode = tn.Node(H)
        self.top_nodes = [self.Hnode]
        self.down_nodes = [self.Hnode]
        for idx, t in enumerate(tidx):
            self.top_edges[t] = [0, idx]
            self.down_edges[t] = [0, len(tidx)+idx]

    
    def prepare_renormalize(self):
        node_list = tn.replicate_nodes(self.top_nodes + self.down_nodes[1:])

        # output edge of renormalized Hamiltonian
        output_edge_order = []
        for i in range(self.n):
            if self.top_edges[i] is not None:
                output_edge_order.append(node_list[self.top_edges[i][0]][self.top_edges[i][1]])
        for i in range(self.n):
            if self.down_edges[i] is not None:
                # if 0, then Hamiltonian
                if self.down_edges[i][0] == 0:
                    output_edge_order.append(node_list[0][self.down_edges[i][1]])
                else:
                    output_edge_order.append(node_list[len(self.top_nodes)+self.down_edges[i][0]-1][self.down_edges[i][1]])
        
        return node_list, output_edge_order

    
    def find_renormalize_tree(self, algorithm=None, seq="", visualize=False):
        """compute contraction tree of renormalization

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_renormalize()

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)


    def renormalize(self, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """renormalize MERA

        Args:
            algorithm : the algorithm to find contraction path
        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_renormalize()
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)

    def gen_renormalize_func(self, tree, backend="jax"):
        """generate renormalization function

        Args:
            tree (ctg.ContractionTree) : the contraction tree for renormalization
            gpunum (int) : the number of GPU we use
        Returns:
            func : get sliced arrays and return renormalized hamiltonian, args (H, U1, U2, ..., U1.conj(), U2.conj(), ...)
        """

        return functools.partial(tree.contract_core, backend=backend)
    
    def visualize_renormalization(self, tn, tree):
        """calc contraction cost and visualize contract path for given tree and nodes

        Args:
            tn (TensorNetwork) : the tensor network
            tree (ctg.ContractionTree) : the contraction tree
        """

        node_list, output_edge_order = self.prepare_renormalize()

        #tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        print(f"contraction cost: {tree.contraction_cost():,} peak size: {tree.peak_size():,}")
        print(tree.get_ssa_path())
        print(tree.flat_tree())
        tree.print_contractions()

    def apply_isometry(self, input_support, output_support, tensor):
        """ apply Unitary or Isometry
        
        Args:
            input_support (list of int) : list of qubit index we apply to.
            output_support (list of int) : list of qubit index of output.
            tensor (np.array) : the tensor of Unitary or Isometry.  shape:(input,...,input,output,...,output)
        """
        Unode = tn.Node(tensor)
        Adnode = tn.Node(tensor.conj())
        self.top_nodes.append(Unode)
        self.down_nodes.append(Adnode)
        # cancel isometry
        for idx, tidx in enumerate(input_support):
            if self.top_edges[tidx] is not None:
                break
            if idx == len(input_support)-1:
                # apply nothing
                return
                
        # connect edges
        for idx, tidx in enumerate(input_support):
            if self.top_edges[tidx] is None:
                # connect themselves
                tn.connect(Unode[idx], Adnode[idx])
            else:
                tn.connect(Unode[idx], self.top_nodes[self.top_edges[tidx][0]][self.top_edges[tidx][1]])
                tn.connect(Adnode[idx], self.down_nodes[self.down_edges[tidx][0]][self.down_edges[tidx][1]])
        
        for tidx in input_support:
            self.top_edges[tidx] = None
            self.down_edges[tidx] = None
        
        for idx, tidx in enumerate(output_support):
            self.top_edges[tidx] = [len(self.top_nodes)-1, len(input_support)+idx]
            self.down_edges[tidx] = [len(self.down_nodes)-1, len(input_support)+idx]

    
    