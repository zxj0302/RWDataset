import scipy.io as scio
from torch_geometric.data import Data
import torch
import os
from tqdm import tqdm

data = scio.loadmat('download/ESC-GNN/data.mat')['A'][0]

def adj_to_data(adj_matrix):
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
    edge_index = adj_tensor.nonzero(as_tuple=False).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=adj_matrix.shape[0], num_edges=edge_index.size(1))
    return data
data_list = [adj_to_data(matrix) for matrix in data]

# use EVOKE to compute ground truth
# 2.1 generate edge_list files EVOKE needs 
# edgelist_graph_path = 'evoke/wrappers/edgelist_graph'
# if not os.path.exists(edgelist_graph_path):
#     os.mkdir(edgelist_graph_path)
# for i, g in tqdm(enumerate(data_list), total=len(data_list), desc='Generating edge list files'):
#     with open(f'{edgelist_graph_path}/rw_{i}.edgelist', 'w') as f:
#         f.write(f'{g.num_nodes} {g.num_edges//2}\n')
#         for edge in g.edge_index.T:
#             if edge[0] < edge[1]:
#                 f.write(f'{edge[0]} {edge[1]}\n')
# print('Edge list files generated')

# 2.2 compute the ground truth using 'run_ego.py' with EVOKE
# output_evoke_path = 'evoke/wrappers/output_evoke'
output_graph_path = 'evoke/wrappers/output_graph'
# evoke_wrappers_path = 'evoke/wrappers/'
# run_evoke_name = 'run_evoke.py'
# if not os.path.exists(output_evoke_path):
#     os.mkdir(output_evoke_path)
# if not os.path.exists(output_graph_path):
#     os.mkdir(output_graph_path)
# # run the EVOKE to compute the ground truth
# os.system(f'cd {evoke_wrappers_path} && python {run_evoke_name} graph')
# # read in the ground truth and add gt_induced_le5 & gt_noninduced_le5 to the graphs
for i, g in tqdm(enumerate(data_list), desc='Adding ground truth to graphs', total=len(data_list)):
    induced = torch.load(f'{output_graph_path}/induced_rw_{i}.pt')
    g.gt_induced_le5 = induced

torch.save(data_list, 'gt_esc.pt')

