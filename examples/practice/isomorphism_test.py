# %%
# import
import random
from copy import deepcopy
from tqdm import tqdm

#%%
# graph generation functions
def make_k_edges(g, max_k):
    node_ids = []
    for i in range(len(g)):
        node_ids.append(i)
    for i in range(max_k):
        end_nodes = random.sample(node_ids, k = 2)
        if not end_nodes[1] in g[end_nodes[0]]['n']:
            g[end_nodes[0]]['n'].append(end_nodes[1])
            g[end_nodes[1]]['n'].append(end_nodes[0])
    return g

def gen_random_g(n: int = 5, max_k: int = 5):
    g = {}
    for i in range(n):
        g[i] = {'n':[], 'L':[], 'c0':1, 'c1':None}
    g = make_k_edges(g, max_k)
    return g

def swap_node_id(g, times: int = 1):
    g = deepcopy(g)
    node_ids = []
    for i in range(len(g)):
        node_ids.append(i)
    for i in range(times):
        nodes = random.sample(node_ids, k = 2)
        # print(nodes)
        g[nodes[0]], g[nodes[1]] = g[nodes[1]], g[nodes[0]]
        for id in node_ids:
            g[id]['n'] = [-1 if id == nodes[0] else id for id in g[id]['n']]
            g[id]['n'] = [nodes[0] if id == nodes[1] else id for id in g[id]['n']]
            g[id]['n'] = [nodes[1] if id == -1 else id for id in g[id]['n']]
    return g
     
# %%
# Weisfeiler Lehman Isomorphism Test
def build_compressed_node_labels(g):
    g = deepcopy(g)
    for i in range(len(g)):
        for node_id in g.keys():
            g[node_id]['L'] = []
            for neighbor_id in g[node_id]['n']:
                g[node_id]['L'].append(g[neighbor_id]['c0'])
            g[node_id]['c1'] = sum(g[node_id]['L']) #TODO
        for node_id in g.keys():
            g[node_id]['c0'] = g[node_id]['c1']
    return g

def areSame(g0, g1):
    if len(g0) != len(g1):
        return False
    g0 = build_compressed_node_labels(g0)
    g1 = build_compressed_node_labels(g1)
    a, b = [], []
    for i in range(len(g0)):
        a.append(g0[i]['c0'])
        b.append(g1[i]['c0'])
    a.sort()
    b.sort()
    return (a == b)

# %%
# bench
error_counter = 0
test_num = 10000
k = 1

with tqdm(total=test_num) as pbar:
    for i in range(test_num):
        # generate g0
        g0 = gen_random_g(50, 30)

        # clone g0, and swap node id
        g1 = swap_node_id(g0, 10)

        # iso test
        iso = areSame(g0, g1)

        # make additional k edges
        g1 = make_k_edges(g1, k)

        # iso test
        iso_mod = areSame(g0, g1)
        
        if iso_mod: error_counter += 1
        
        pbar.update(1)

print(error_counter/test_num)

# %%
