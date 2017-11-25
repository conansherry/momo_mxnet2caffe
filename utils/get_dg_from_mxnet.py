import networkx as nx
import matplotlib.pyplot as plt
import json
import logging
logging.basicConfig(level=logging.DEBUG)

def looks_like_weight(name):
    """Internal helper to figure out if node should be hidden with `hide_weights`.
    """
    if name.endswith("_weight"):
        return True
    if name.endswith("_bias"):
        return True
    if name.endswith("_beta") or name.endswith("_gamma") or name.endswith("_moving_var") or name.endswith(
            "_moving_mean"):
        return True
    return False

def get_dg_from_mxnet(symbol):
    json_symbol = json.loads(symbol.tojson())
    nodes = json_symbol['nodes']

    DG = nx.DiGraph()

    # all nodes
    weight_nodes = []
    input_nodes = []
    all_nodes = []
    attr_dict = symbol.attr_dict()
    op_dict = dict()
    for node in nodes:
        op = node['op']
        name = node['name']
        op_dict[name] = op
        if op == 'null':
            if looks_like_weight(name):
                weight_nodes.append(name)
                continue
            else:
                input_nodes.append(name)
        all_nodes.append(node)
        DG.add_node(name)

    # all edges
    edges_from_to = []
    inputs_map = dict()
    for node in nodes:
        op = node['op']
        name = node['name']
        if op == "null":
            continue
        else:
            inputs = node['inputs']
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node['name']
                if input_name not in weight_nodes:
                    edges_from_to.append((input_name, name))
            inputs_map[name] = inputs
    DG.add_edges_from(edges_from_to)

    sorted_nodes = list(nx.topological_sort(DG))
    edges_dict = dict(edges_from_to)

    return input_nodes, sorted_nodes, edges_dict, attr_dict, op_dict, weight_nodes, inputs_map, all_nodes
