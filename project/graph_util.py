from networkx.drawing import nx_pydot as pydot
from networkx import MultiDiGraph as mdg
from typing import Tuple as tup
import cfpq_data as c_d
from collections import namedtuple

GraphInfo = namedtuple('GraphInfo', 'node_count edge_count labels')


def load(name):
    return c_d.graph_from_csv(c_d.download(name))


def get_info(graph):
    return GraphInfo(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        set([x for _, _, x in graph.edges(data="label")]),
    )


def make_cycles(n, m, labels, path):
    graph = c_d.labeled_two_cycles_graph(n, m, labels=labels)
    pydot.write_dot(graph, path)
