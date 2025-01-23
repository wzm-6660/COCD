import dgl


def build_graph(type, node):
    g = dgl.DGLGraph()
    g.add_nodes(node)
    edge_list = []

    if type == 'coograph':
        with open('../data/EdNet-1/graph/cooccurrenceGraph.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        edge_list = list(set(edge_list))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        return g
    elif type == 'rcoograph':
        with open('../data/EdNet-1/graph/relatedCooccurrenceGraph.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        edge_list = list(set(edge_list))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)
        return g
    elif type == 'k_from_e':
        with open('../data/EdNet-1/graph/k_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open('../data/EdNet-1/graph/e_from_k.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'u_from_e':
        with open('../data/EdNet-1/graph/u_from_e.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_u':
        with open('../data/EdNet-1/graph/e_from_u.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'direct':
        with open('../data/EdNet-1/graph/K_Directed.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        # g.add_edges(dst, src)
        return g
    elif type == 'undirect':
        with open('../data/EdNet-1/graph/K_Undirected.txt', 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        # edges are directional in DGL; make them bi-directional
        g.add_edges(dst, src)
        return g
