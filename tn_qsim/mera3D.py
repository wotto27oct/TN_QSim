import numpy as np
from numpy.core.fromnumeric import reshape
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork
from tn_qsim.mpo import MPO
from tn_qsim.utils import from_tn_to_quimb, from_nodes_to_str

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

    def __init__(self, n, H, tidx, name=None):
        self.n = n
        self.top_edges = [None for _ in range(n)]
        self.down_edges = [None for _ in range(n)]
        self.Hnode = tn.Node(H, name=name)
        self.top_nodes = [self.Hnode]
        self.down_nodes = [self.Hnode]
        for idx, t in enumerate(tidx):
            self.top_edges[t] = [0, idx]
            self.down_edges[t] = [0, len(tidx)+idx]

    def prepare_renormalize(self, rho, rhosupp):
        node_list = tn.replicate_nodes(self.top_nodes + self.down_nodes[1:])

        # output edge of renormalized Hamiltonian
        """output_edge_order = []
        for i in range(self.n):
            if self.top_edges[i] is not None:
                output_edge_order.append(node_list[self.top_edges[i][0]][self.top_edges[i][1]])
        for i in range(self.n):
            if self.down_edges[i] is not None:
                # if 0, then Hamiltonian
                if self.down_edges[i][0] == 0:
                    output_edge_order.append(node_list[0][self.down_edges[i][1]])
                else:
                    output_edge_order.append(node_list[len(self.top_nodes)+self.down_edges[i][0]-1][self.down_edges[i][1]])"""
        
        output_edge_order = []
        identity_node_list = []
        input_rho_node = tn.Node(rho)
        for idx, tidx in enumerate(rhosupp):
            if self.top_edges[tidx] is not None:
                output_edge_order.append(node_list[self.top_edges[tidx][0]][self.top_edges[tidx][1]])
            else:
                identity_node_list.append(tn.Node(np.eye(rho.shape[idx]), name="I"))
                output_edge_order.append(identity_node_list[-1][0])
        identity_node_idx = 0
        for idx, tidx in enumerate(rhosupp):
            if self.top_edges[tidx] is not None:
                # if 0, then Hamiltonian
                if self.down_edges[tidx][0] == 0:
                    output_edge_order.append(node_list[0][self.down_edges[tidx][1]])
                else:
                    output_edge_order.append(node_list[len(self.top_nodes)+self.down_edges[tidx][0]-1][self.down_edges[tidx][1]])
            else:
                output_edge_order.append(identity_node_list[identity_node_idx][1])
                identity_node_idx += 1
        
        node_list += identity_node_list
        
        return node_list, output_edge_order
    
    def find_renormalize_tree(self, rho, rhosupp, algorithm=None, seq="", visualize=False):
        """compute contraction tree of renormalization

        Args:
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_renormalize(rho, rhosupp)

        input_name = []
        for node in node_list:
            input_name.append(node.name)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

        #if visualize:
        #    node_list, output_edge_order = self.prepare_renormalize()
        #    self.visualize_tree(tree, node_list, output_edge_order, visualize=True)
        
        return tn, tree, input_name

    def renormalize(self, rho, rhosupp, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=""):
        """renormalize MERA

        Args:
            algorithm : the algorithm to find contraction path
        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_renormalize(rho, rhosupp)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)

    def prepare_inv_renormalize(self, rho, rhosupp, name):
        node_list = tn.replicate_nodes(self.top_nodes + self.down_nodes[1:])

        # input edge of renormalized Hamiltonian
        """input_edge_order = []
        for i in range(self.n):
            if self.top_edges[i] is not None:
                input_edge_order.append(node_list[self.top_edges[i][0]][self.top_edges[i][1]])
        for i in range(self.n):
            if self.down_edges[i] is not None:
                # if 0, then Hamiltonian
                if self.down_edges[i][0] == 0:
                    input_edge_order.append(node_list[0][self.down_edges[i][1]])
                else:
                    input_edge_order.append(node_list[len(self.top_nodes)+self.down_edges[i][0]-1][self.down_edges[i][1]])
                    
        input_rho_node = tn.Node(rho, name=name)
        for i in range(len(input_edge_order)):
            tn.connect(input_rho_node[i], input_edge_order[i])"""

        input_rho_node = tn.Node(rho, name=name)
        for idx, tidx in enumerate(rhosupp):
            if self.top_edges[tidx] is not None:
                tn.connect(node_list[self.top_edges[tidx][0]][self.top_edges[tidx][1]], input_rho_node[idx])
                # if 0, then Hamiltonian
                if self.down_edges[tidx][0] == 0:
                    tn.connect(node_list[0][self.down_edges[tidx][1]], input_rho_node[idx+len(rhosupp)])
                else:
                    tn.connect(node_list[len(self.top_nodes)+self.down_edges[tidx][0]-1][self.down_edges[tidx][1]], input_rho_node[idx+len(rhosupp)])
            else:
                # no tensors between rho_iI
                tn.connect(input_rho_node[idx], input_rho_node[idx+len(rhosupp)])   
        
        # output edge is original hamiltonian
        output_edge_order = node_list[0].edges

        node_list = [input_rho_node] + node_list[1:]
        
        return node_list, output_edge_order

    def find_inv_renormalize_tree(self, rho, rhosupp, name=None, algorithm=None, seq="", visualize=False):
        """compute contraction tree of inverse renormalization

        Args:
            rho (np.array) : the input reduced density op to be inv-renormalized
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_inv_renormalize(rho, rhosupp, name)

        input_name = []
        for node in node_list:
            input_name.append(node.name)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

        return tn, tree, input_name

    def inv_renormalize(self, rho, rhosupp, name=None, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """inv-renormalize MERA

        Args:
            algorithm : the algorithm to find contraction path
        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_inv_renormalize(rho, rhosupp, name)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)
    
    def prepare_energy(self, rho, rhosupp, name):
        node_list = tn.replicate_nodes(self.top_nodes + self.down_nodes[1:])

        # input edge of renormalized Hamiltonian
        """input_edge_order = []
        for i in range(self.n):
            if self.top_edges[i] is not None:
                input_edge_order.append(node_list[self.top_edges[i][0]][self.top_edges[i][1]])
        for i in range(self.n):
            if self.down_edges[i] is not None:
                # if 0, then Hamiltonian
                if self.down_edges[i][0] == 0:
                    input_edge_order.append(node_list[0][self.down_edges[i][1]])
                else:
                    input_edge_order.append(node_list[len(self.top_nodes)+self.down_edges[i][0]-1][self.down_edges[i][1]])

        input_rho_node = tn.Node(rho, name=name)
        for i in range(len(input_edge_order)):
            tn.connect(input_rho_node[i], input_edge_order[i])"""
        
        input_rho_node = tn.Node(rho, name=name)
        for idx, tidx in enumerate(rhosupp):
            if self.top_edges[tidx] is not None:
                tn.connect(node_list[self.top_edges[tidx][0]][self.top_edges[tidx][1]], input_rho_node[idx])
                # if 0, then Hamiltonian
                if self.down_edges[tidx][0] == 0:
                    tn.connect(node_list[0][self.down_edges[tidx][1]], input_rho_node[idx+len(rhosupp)])
                else:
                    tn.connect(node_list[len(self.top_nodes)+self.down_edges[tidx][0]-1][self.down_edges[tidx][1]], input_rho_node[idx+len(rhosupp)])
            else:
                # no tensors between rho_iI
                tn.connect(input_rho_node[idx], input_rho_node[idx+len(rhosupp)])
        
        # no output edge
        output_edge_order = []

        node_list = [input_rho_node] + node_list

        return node_list, output_edge_order

    def find_energy_tree(self, rho, rhosupp, name=None, algorithm=None, seq="", visualize=False):
        """compute contraction tree of inverse renormalization

        Args:
            rho (np.array) : the input reduced density op to be inv-renormalized
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_energy(rho, rhosupp, name)

        input_name = []
        for node in node_list:
            input_name.append(node.name)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        tn, tree = self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)

        return tn, tree, input_name

    def energy(self, rho, rhosupp, name=None, algorithm=None, tn=None, tree=None, target_size=None, gpu=True, thread=1, seq=None):
        """calculate MERA energy given renormalized density operator

        Args:
            algorithm : the algorithm to find contraction path
        Returns:
            np.array: tensor after contraction
        """

        if tn is None:
            node_list, output_edge_order = self.prepare_energy(rho, rhosupp, name)
            tn, _ = from_tn_to_quimb(node_list, output_edge_order)

        return self.contract_tree_by_quimb(tn, algorithm, tree, None, target_size, gpu, thread, seq)

    def prepare_grad(self, rho, idx):
        """prepare grad for idx-th unitary/isometry given renormalized hamiltonian and rho
        """
        node_list = tn.replicate_nodes(self.top_nodes + self.down_nodes[1:])

        # output edge of renormalized Hamiltonian
        input_edge_order = []
        for i in range(self.n):
            if self.top_edges[i] is not None:
                input_edge_order.append(node_list[self.top_edges[i][0]][self.top_edges[i][1]])
        for i in range(self.n):
            if self.down_edges[i] is not None:
                # if 0, then Hamiltonian
                if self.down_edges[i][0] == 0:
                    input_edge_order.append(node_list[0][self.down_edges[i][1]])
                else:
                    input_edge_order.append(node_list[len(self.top_nodes)+self.down_edges[i][0]-1][self.down_edges[i][1]])
            
        input_rho_node = tn.Node(rho)
        for i in range(len(input_edge_order)):
            tn.connect(input_rho_node[i], input_edge_order[i])

        # output edge is idx-th isometry/unitary
        # 0th is hamiltonian
        output_edge_order = node_list[idx+1].edges
        node_list.pop(idx+1)

        node_list = [input_rho_node] + node_list

        return node_list, output_edge_order

    def find_grad_tree(self, rho, idx, algorithm=None, seq="", visualize=False):
        """compute contraction tree of grad-calculation for idx

        Args:
            rho (np.array) : the input reduced density op to be inv-renormalized
            idx (int) : the unitary/isometry index to be popped, 0 start
            algorithm : the algorithm to find contraction path

        Returns:
            tn (TensorNetwork) : tn for contract
            tree (ContractionTree) : contraction tree for contract
        """

        node_list, output_edge_order = self.prepare_grad(rho, idx)

        tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        if visualize:
            print(f"before simplification  |V|: {tn.num_tensors}, |E|: {tn.num_indices}")

        return self.find_contract_tree_by_quimb(tn, output_inds, algorithm, seq=seq)
    
    def visualize_renormalization(self, tn, tree):
        """calc contraction cost and visualize contract path for given tree and nodes

        Args:
            tn (TensorNetwork) : the tensor network
            tree (ctg.ContractionTree) : the contraction tree
        """

        node_list, output_edge_order = self.prepare_renormalize()
        input_alpha, output_alpha, edge_alpha_dims = from_nodes_to_str(node_list, output_edge_order, offset=8)

        #tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        print(f"contraction cost: {tree.contraction_cost():,} peak size: {tree.peak_size():,}")
        print("inputs:", input_alpha)
        print("outputs:", output_alpha)
        print(f"slice: {tree.sliced_inds}")
        print(tree.get_ssa_path())
        print(tree.flat_tree())
        tree.print_contractions()

    def visualize_inv_renormalization(self, tn, tree, rhotmp):
        """calc contraction cost and visualize contract path for given tree and nodes

        Args:
            tn (TensorNetwork) : the tensor network
            tree (ctg.ContractionTree) : the contraction tree
        """

        node_list, output_edge_order = self.prepare_inv_renormalize(rhotmp)
        input_alpha, output_alpha, edge_alpha_dims = from_nodes_to_str(node_list, output_edge_order, offset=8)

        #tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        print(f"contraction cost: {tree.contraction_cost():,} peak size: {tree.peak_size():,}")
        print("inputs:", input_alpha)
        print("outputs:", output_alpha)
        print("dims:", edge_alpha_dims)
        print(f"slice: {tree.sliced_inds}")
        print(tree.get_ssa_path())
        print(tree.flat_tree())
        tree.print_contractions()

    def visualize_energy(self, tn, tree, rhotmp):
        """calc contraction cost and visualize contract path for given tree and nodes

        Args:
            tn (TensorNetwork) : the tensor network
            tree (ctg.ContractionTree) : the contraction tree
        """

        node_list, output_edge_order = self.prepare_energy(rhotmp)
        input_alpha, output_alpha, edge_alpha_dims = from_nodes_to_str(node_list, output_edge_order, offset=8)

        #tn, output_inds = from_tn_to_quimb(node_list, output_edge_order)

        print(f"contraction cost: {tree.contraction_cost():,} peak size: {tree.peak_size():,}")
        print("inputs:", input_alpha)
        print("outputs:", output_alpha)
        print(f"slice: {tree.sliced_inds}")
        print(tree.get_ssa_path())
        print(tree.flat_tree())
        tree.print_contractions()

    def apply_isometry(self, input_support, output_support, tensor, name=None):
        """ apply Unitary or Isometry
        
        Args:
            input_support (list of int) : list of qubit index we apply to.
            output_support (list of int) : list of qubit index of output.
            tensor (np.array) : the tensor of Unitary or Isometry.  shape:(input,...,input,output,...,output)
        """
        Unode = tn.Node(tensor, name=name)
        Adnode = tn.Node(tensor.conj(), name=name+"c")
        # cancel isometry
        for idx, tidx in enumerate(input_support):
            if self.top_edges[tidx] is not None:
                break
            if idx == len(input_support)-1:
                # apply nothing
                return
        self.top_nodes.append(Unode)
        self.down_nodes.append(Adnode)
                
        # connect edges
        #print(input_support)
        for idx, tidx in enumerate(input_support):
            if self.top_edges[tidx] is None:
                # connect themselves
                tn.connect(Unode[idx], Adnode[idx])
            else:
                #print(idx, tidx)
                tn.connect(Unode[idx], self.top_nodes[self.top_edges[tidx][0]][self.top_edges[tidx][1]])
                tn.connect(Adnode[idx], self.down_nodes[self.down_edges[tidx][0]][self.down_edges[tidx][1]])
        
        for tidx in input_support:
            self.top_edges[tidx] = None
            self.down_edges[tidx] = None
        
        for idx, tidx in enumerate(output_support):
            self.top_edges[tidx] = [len(self.top_nodes)-1, len(input_support)+idx]
            self.down_edges[tidx] = [len(self.down_nodes)-1, len(input_support)+idx]