import opt_einsum as oe
import tensornetwork as tn
import quimb.tensor as qtn

def from_nodes_to_str(node_list, output_edge_order):
    input_sets = [set(node.edges) for node in node_list]
    output_set = set()
    for edge in tn.get_all_edges(node_list):
        if edge.is_dangling() or not set(edge.get_nodes()) <= set(node_list):
            output_set.add(edge)
    size_dict = {edge: edge.dimension for edge in tn.get_all_edges(node_list)}

    edge_alpha = dict()
    edge_alpha_dims = dict()
    alpha_offset = 0
    for node in node_list:
        for e in node.edges:
            if not e in edge_alpha:
                edge_alpha[e] = oe.get_symbol(alpha_offset)
                alpha_offset += 1
                edge_alpha_dims[edge_alpha[e]] = size_dict[e]

    input_alpha = []
    for node in node_list:
        str = []
        for e in node.edges:
            str.append(edge_alpha[e])
        input_alpha.append(str)

    output_alpha = None
    if output_edge_order is not None:
        str = []
        for e in output_edge_order:
            str.append(edge_alpha[e])
        output_alpha = str
    else:
        str = []
        for e in output_set:
            str.append(edge_alpha[e])
        output_alpha = str
    
    return input_alpha, output_alpha, edge_alpha_dims


def from_tn_to_quimb(node_list, output_edge_order):
    input_alpha, output_alpha, edge_alpha_dims = from_nodes_to_str(node_list, output_edge_order)
    tensors = []
    for idx, node in enumerate(node_list):
        tensors.append(qtn.Tensor(data=node.tensor, inds=list(input_alpha[idx])))
        node.tensor = None
    tn = qtn.TensorNetwork(tensors)
    output_alpha = "".join(output_alpha)
    return tn, output_alpha