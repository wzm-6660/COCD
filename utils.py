import argparse
from GCCD.build_graph import build_graph

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()
        self.add_argument('--exer_n', type=int, default=11996)
        self.add_argument('--knowledge_n', type=int, default=188)
        self.add_argument('--student_n', type=int, default=1827)
        self.add_argument('--gpu', type=int, default=0)
        self.add_argument('--epoch_n', type=int, default=25)
        self.add_argument('--test', action='store_true')
        self.add_argument('--lr', type=float, default=0.0005)
        self.add_argument('--batch_size', type=int, default=128)
        # self.add_argument('--gclhidden_dim', type=int, default=32)
        # self.add_argument('--gclnum_layers', type=int, default=2)
        self.add_argument('--value_range', type=int, default=1)
        self.add_argument('--a_range', type=int, default=1)
        self.add_argument('--latent_dim', type=int, default=5)

def construct_local_map(args):
    local_map = {
        
        'directed_g': build_graph('direct', args.knowledge_n),
        'undirected_g': build_graph('undirect', args.knowledge_n),
        'k_from_e': build_graph('k_from_e', args.knowledge_n + args.exer_n),
        'e_from_k': build_graph('e_from_k', args.knowledge_n + args.exer_n),
        'u_from_e': build_graph('u_from_e', args.student_n + args.exer_n),
        'e_from_u': build_graph('e_from_u', args.student_n + args.exer_n),
        'coograph': build_graph('coograph',args.knowledge_n),
        'rcoograph': build_graph('rcoograph',args.knowledge_n),
    }
    return local_map

