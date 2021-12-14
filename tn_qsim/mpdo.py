import numpy as np
import opt_einsum as oe
import tensornetwork as tn
from tn_qsim.general_tn import TensorNetwork

class MPDO(TensorNetwork):
    """class of MPDO

    physical bond (up) 0, 1, ..., n-1
    virtual bond: n, n+1, ..., 2n
    inner bond (down) 2n+1, ..., 3n

    Attributes:
        n (int) : the number of tensors
        apex (int) : apex point of canonical form
        edges (list of list of int) : the orderd indexes of each edge connected to each tensor
        edge_dims (dict of int) : dims of each edges
        tensors (list of np.array) : each tensor, [physical(up), inner(down), virtual_left, virtual_right]
        truncate_dim (int) : truncation dim of virtual bond, default None
    """

    def __init__(self, tensors, truncate_dim=None, threthold_err=None):
        self.n = len(tensors)
        edge_info = []
        for i in range(self.n):
            edge_info.append([i, self.n+i, self.n+i+1, 2*self.n+i+1])
        super().__init__(edge_info, tensors)
        self.apex = None
        self.truncate_dim = truncate_dim
        self.threthold_err = threthold_err


    @property
    def virtual_dims(self):
        virtual_dims = [self.nodes[0].get_dimension(1)]
        for i in range(self.n):
            virtual_dims.append(self.nodes[i].get_dimension(2))
        return virtual_dims


    @property
    def inner_dims(self):
        inner_dims = []
        for i in range(self.n):
            inner_dims.append(self.nodes[i].get_dimension(3))
        return inner_dims


    def canonicalization(self):
        """canonicalize MPDO
        apex point = self.0

        """
        self.apex = 0
        for i in range(self.n-1):
            self.__move_right_canonical()
        for i in range(self.n-1):
            self.__move_left_canonical()


    def contract(self):
        """contract and generate density operator.

        conjugate tensor is appended.

        all edges which dim is 1 is excluded.
        
        Returns:
            np.array: tensor after contraction
        """
        cp_nodes = tn.replicate_nodes(self.nodes)
        for i in range(self.n):
            cp_nodes.append(tn.Node(cp_nodes[i].tensor.conj()))
            tn.connect(cp_nodes[i][3], cp_nodes[i+self.n][3])
            if i != 0:
                tn.connect(cp_nodes[self.n+i-1][2], cp_nodes[self.n+i][1])
        
        output_edge_order = [cp_nodes[0][1], cp_nodes[self.n][1], cp_nodes[self.n-1][2], cp_nodes[2*self.n-1][2]]
        for i in range(self.n):
            output_edge_order.append(cp_nodes[i][0])
        for i in range(self.n):
            output_edge_order.append(cp_nodes[self.n+i][0])
        return tn.contractors.auto(cp_nodes, output_edge_order=output_edge_order).tensor

    def apply_MPO(self, tidx, mpo, is_normalize=True):
        """ apply MPO

        Args:
            tidx (list of int) : list of qubit index we apply to.
            mpo (MPO) : MPO tensornetwork.
        """

        # apexをtidx[0]に合わせる
        if self.apex is not None:
            if tidx[0] < self.apex:
                for _ in range(self.apex - tidx[0]):
                    self.__move_left_canonical()
            elif tidx[0] > self.apex:
                for _ in range(tidx[0] - self.apex):
                    self.__move_right_canonical()
    
        is_direction_right = False
        if len(tidx) == 1:
            is_direction_right = True
        else:
            if tidx[1] - tidx[0] == 1:
                is_direction_right = True
        for i in range(len(tidx)-1):
            if is_direction_right and tidx[i+1] - tidx[i] != 1 or not is_direction_right and tidx[i+1] - tidx[i] != -1:
                raise ValueError("gate must be applied in sequential to MPS")
        
        dir = 2 if is_direction_right else 1

        total_fidelity = 1.0
            
        edge_list = []
        node_list = []
        if len(tidx) == 1:
            node = mpo.nodes[0]
            node_contract_list = [node, self.nodes[tidx[0]]]
            node_edge_list = [node[0]] + [self.nodes[tidx[0]][j] for j in range(1, 4)] + [node[3]]
            one = tn.Node(np.array([1]))
            tn.connect(node[2], one[0])
            node_contract_list.append(one)
            tn.connect(node[1], self.nodes[tidx[0]][0])
            new_node = tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list)
            tn.flatten_edges([new_node[3], new_node[4]])
            node_list.append(new_node)
        else:
            for i, node in enumerate(mpo.nodes):
                if i == 0:
                    node_contract_list = [node, self.nodes[tidx[i]]]
                    node_edge_list = [node[0]] + [self.nodes[tidx[i]][j] for j in range(1, 4)] + [node[3]]
                    one = tn.Node(np.array([1]))
                    tn.connect(node[2], one[0])
                    node_contract_list.append(one)
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    node_list.append(tn.contractors.auto(node_contract_list, output_edge_order=node_edge_list))
                    edge_list.append(node_edge_list)
                else:
                    tn.connect(node[1], self.nodes[tidx[i]][0])
                    l_l_edges = [node_list[i-1][j] for j in range(0, 4) if j != dir]
                    l_r_edges = [node_list[i-1][dir]] + [node_list[i-1][4]]
                    lU, ls, lVh, _ = tn.split_node_full_svd(node_list[i-1], l_l_edges, l_r_edges)
                    lU = lU.reorder_edges(l_l_edges + [ls[0]])
                    lVh = lVh.reorder_edges(l_r_edges + [ls[1]])
                    r_l_edges = [self.nodes[tidx[i]][0]] + [self.nodes[tidx[i]][(dir%2)+1]]
                    r_r_edges = [self.nodes[tidx[i]][dir]] + [self.nodes[tidx[i]][3]]
                    rU, rs, rVh, _ = tn.split_node_full_svd(self.nodes[tidx[i]], r_l_edges, r_r_edges)
                    rU = rU.reorder_edges(r_l_edges + [rs[0]])
                    rVh = rVh.reorder_edges(r_r_edges + [rs[1]])
                    svd_node_list = [ls, lVh, rU, rs, node]
                    svd_node_edge_list = [ls[0], node[0], node[3], rs[1]]
                    svd_node = tn.contractors.optimal(svd_node_list, output_edge_order=svd_node_edge_list)
                    U, s, Vh, trun_s = tn.split_node_full_svd(svd_node, [svd_node[0]], [svd_node[i] for i in range(1, len(svd_node.edges))], self.truncate_dim)
                    s_sq = np.dot(np.diag(s.tensor), np.diag(s.tensor))
                    trun_s_sq = np.dot(trun_s, trun_s)
                    fidelity = s_sq / (s_sq + trun_s_sq)
                    total_fidelity *= fidelity
                    l_edge_order = [lU.edges[i] for i in range(0, dir)] + [s[0]] + [lU.edges[i] for i in range(dir, 3)]
                    node_list[i-1] = tn.contractors.optimal([lU, U], output_edge_order=l_edge_order)
                    r_edge_order = [Vh[1]] + [rVh.edges[i] for i in range(0, (dir%2))] + [s[0]] + [rVh.edges[i] for i in range(0, (dir+1)%2)] + [rVh.edges[1]] + [Vh[2]]
                    new_node = tn.contractors.optimal([s, Vh, rVh], output_edge_order=r_edge_order)
                    if i == mpo.n - 1:
                        tn.flatten_edges([new_node[3], new_node[4]])
                    node_list.append(new_node)

        for i in range(len(tidx)):
            self.nodes[tidx[i]] = node_list[i]

        # heuristic simple-update for inner dimension
        shape_list = self.nodes[tidx[-1]].tensor.shape
        if shape_list[0] * shape_list[1] * shape_list[2] < shape_list[3]:
            tmp = oe.contract("abcd,efgd->abcefg", self.nodes[tidx[-1]].tensor, self.nodes[tidx[-1]].tensor.conj())
            U, s, Vh = np.linalg.svd(tmp.reshape(shape_list[0]*shape_list[1]*shape_list[2], -1), full_matrices=False)
            self.nodes[tidx[-1]].set_tensor(oe.contract("ab,bc->ac", U, np.diag(np.sqrt(s))).reshape(shape_list[0], shape_list[1], shape_list[2], -1))

        if self.apex is not None:
            self.apex = tidx[-1]
        if is_normalize:
            self.nodes[tidx[-1]].tensor = self.nodes[tidx[-1]].tensor / np.sqrt(self.calc_trace().flatten()[0])
        
        return total_fidelity


    def sample(self, seed=0):
        """ sample from mpdo
            not implemented yet
        """
        #for _ in range(self.apex, 0, -1):
        #    self.__move_left_canonical()

        np.random.seed(seed)

        output = []
        left_tensor = np.array([1]).reshape(1,1)
        #for i in range(self.n-1):
        #    left_tensor.append(oe.contract("aacd,c->d", self.tensors[i], left_tensor[i]))
        right_tensor = [np.array([1]).reshape(1,1)]
        for i in range(self.n-1, 0, -1):
            right_tensor.append(oe.contract("abcd,efgd,cg->bf", self.nodes[i].tensor, self.nodes[i].tensor.conj(), right_tensor[self.n-1-i]))
        right_tensor = right_tensor[::-1]
        zero = np.array([1, 0])
        one = np.array([0, 1])
        for i in range(self.n):
            prob_matrix = oe.contract("abcd,efgd,bf,cg->ae", self.nodes[i].tensor, self.nodes[i].tensor.conj(), left_tensor, right_tensor[i])
            rand_val = np.random.uniform()
            if rand_val < prob_matrix[0][0] / np.trace(prob_matrix):
                output.append(0)
                left_tensor = oe.contract("abcd,efgd,bf,a,e->cg", self.nodes[i].tensor, self.nodes[i].tensor.conj(), left_tensor, zero.conj(), zero)
            else:
                output.append(1)
                left_tensor = oe.contract("abcd,efgd,bf,a,e->cg", self.nodes[i].tensor, self.nodes[i].tensor.conj(), left_tensor, one.conj(), one)
        
        return np.array(output)


    def calc_trace(self):
        left_tensor = oe.contract("abcd,aefd->becf", self.nodes[0].tensor, self.nodes[0].tensor.conj())
        for i in range(1, self.n):
            left_tensor = oe.contract("ghbe,abcd,aefd->ghcf", left_tensor, self.nodes[i].tensor, self.nodes[i].tensor.conj())
        return left_tensor


    def __move_right_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == self.n-1:
            raise ValueError("can't move canonical apex to right")
        l_edges = self.nodes[self.apex].get_all_edges()
        r_edges = self.nodes[self.apex+1].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [l_edges[0], l_edges[1], l_edges[3]], [l_edges[2]])
        self.nodes[self.apex] = U.reorder_edges([l_edges[0], l_edges[1], s[0], l_edges[3]])
        self.nodes[self.apex+1] = tn.contractors.optimal([s, Vh, self.nodes[self.apex+1]], output_edge_order=[r_edges[0], s[0], r_edges[2], r_edges[3]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex+1].set_name(f"node {self.apex+1}")
        self.nodes[self.apex][2].set_name(f"edge {self.apex+self.n+1}")

        self.apex = self.apex + 1

    def __move_left_canonical(self):
        """ move canonical apex to right
        """
        if self.apex == 0:
            raise ValueError("can't move canonical apex to left")
        l_edges = self.nodes[self.apex-1].get_all_edges()
        r_edges = self.nodes[self.apex].get_all_edges()
        U, s, Vh, _ = tn.split_node_full_svd(self.nodes[self.apex], [r_edges[1]], [r_edges[0], r_edges[2], r_edges[3]])
        self.nodes[self.apex] = Vh.reorder_edges([r_edges[0], s[1], r_edges[2], r_edges[3]])
        self.nodes[self.apex-1] = tn.contractors.optimal([self.nodes[self.apex-1], U, s], output_edge_order=[l_edges[0], l_edges[1], s[1], l_edges[3]])

        self.nodes[self.apex].set_name(f"node {self.apex}")
        self.nodes[self.apex-1].set_name(f"node {self.apex-1}")
        self.nodes[self.apex][1].set_name(f"edge {self.apex+self.n}")

        self.apex = self.apex - 1