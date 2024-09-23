#%% md
# TODO:
# 1. 模块化print和log， 并可视化, log 2 a file
# 2. 合理规划生成的数据文件的命名
# 3. 查看并行化的bug以及batch_size的设置
# 4. 优化传入的参数
# 5. 修改GIN，看里面用的几层MLP
# 6. settle down all random seeds
# 7. change output to positive integer
# 8. change labels to log scale
# 9. make tqdm runs well with multiprocessing
# 10. figure out why the slp is not updated
# 11. find good ways to visualize the computation graph
#%%
import torch
import torch.nn as nn
import torch_geometric.datasets as datasets
from torch_geometric.nn.models import GIN, MLP
import pytorch_lightning as pl
from pytorch_lightning import loggers
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import StochasticWeightAveraging
from tqdm.notebook import tqdm
import networkx as nx
from torch_geometric.utils import to_networkx
from torch.utils.data import random_split
from torchviz import make_dot
import os
#%%
class ENDataModule(pl.LightningDataModule):
    def __init__(self, dataset_list, queries, train_valid_test=(0.7, 0.2, 0.1), hop=3, batch_s=64, num_workers=8):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_list = dataset_list
        self.query_graph = queries
        self.train_valid_test = train_valid_test
        self.hop = hop
        self.batch_size = batch_s
        self.num_workers = num_workers
        self.dataset_source = {'IMDB-BINARY': 'TUDataset', 'PTC_MR': 'TUDataset', 'PROTEINS': 'TUDataset',
                               'NCI1': 'TUDataset', 'NCI109': 'TUDataset', 'MUTAG': 'TUDataset', 'ENZYMES': 'TUDataset',
                               'DD': 'TUDataset', 'COLLAB': 'TUDataset', 'REDDIT-BINARY': 'TUDataset',
                               'REDDIT-MULTI-5K': 'TUDataset', 'REDDIT-MULTI-12K': 'TUDataset', 'QM9': 'TUDataset',
                               'FIRSTMM_DB': 'TUDataset', 'ZINC': 'TUDataset', 'Cora': 'Planetoid',
                               'CiteSeer': 'Planetoid', 'PubMed': 'Planetoid'}
        # just assume the 'query_graph' contains only one graph and the file_path is not valid on Windows System
        # the query_graph must contain an attribute 'name'
        self.file_path = 'data/ENDataset/' + '&'.join(sorted(dataset_list)) + '@' + '&'.join(sorted(qg.graph['name'] for qg in self.query_graph)) + '.pt'
        self.processed_data, self.raw_data, self.train_dataset, self.val_dataset, self.test_dataset = (None,) * 5

    def prepare_data(self):
        for ds in self.dataset_list:
            if ds not in self.dataset_source:
                raise ValueError(f'Dataset {ds} not included')
            else:
                getattr(datasets, self.dataset_source[ds])(f'data/{self.dataset_source[ds]}', ds)

    def setup(self, stage=None):
        if os.path.exists(self.file_path):
            self.processed_data = torch.load(self.file_path)
        else:
            self.raw_data = [(getattr(datasets, self.dataset_source[ds])(f'data/{self.dataset_source[ds]}', ds)) for ds
                             in self.dataset_list]
            self.dataset_generator()

            # save the dataset to file and print information
            torch.save(self.processed_data, self.file_path)
            print('File saved to ', self.file_path)

        # split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.processed_data,
                                                                               self.train_valid_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, pin_memory=True, drop_last=True)

    def predict_dataloader(self):
        pass

    def dataset_generator(self):
        num_automorphisms = []
        for query_graph_ in self.query_graph:
            num_automorphism_per = 0
            for _ in nx.algorithms.isomorphism.GraphMatcher(query_graph_, query_graph_).subgraph_isomorphisms_iter():
                num_automorphism_per += 1
            num_automorphisms.append(num_automorphism_per)
        print("Number of automorphism of query graphs: ", num_automorphisms)

        data_list, all_tasks = [], []
        for raw in self.raw_data:
            for graph in raw:
                graph = to_networkx(graph, to_undirected=True)
                for node in graph.nodes():
                    ego = nx.ego_graph(graph, node, radius=self.hop)
                    all_tasks.append((ego, node, num_automorphisms))
        print(f'{len(all_tasks)} tasks submitted')

        for task in tqdm(all_tasks, total=len(all_tasks)):
            data_list.append(self.dataset_generator_per(task))

        self.processed_data = data_list

    def dataset_generator_per(self, task):
        ego, central_node, num_automorphisms = task

        # compute the subgraph counts for each query graph
        item_y = []
        for it in range(len(self.query_graph)):
            item_y_per = 0
            for i in nx.algorithms.isomorphism.GraphMatcher(ego, self.query_graph[it]).subgraph_isomorphisms_iter():
                if central_node in i.keys():
                    item_y_per += 1
            item_y.append(item_y_per / num_automorphisms[it])

        # convert the ego network to torch_geometric.data.Data object and set the node 0 as central node
        mapping = {central_node: 0}
        for n in ego.nodes():
            if n != central_node:
                mapping[n] = len(mapping)
        ego = nx.relabel_nodes(ego, mapping)
        # get the edge_index with COO format, be careful with the condition of empty edges
        edges = sorted([(e[0], e[1]) for e in ego.edges()] + [(e[1], e[0]) for e in ego.edges()],
                       key=lambda x: (x[0], x[1]))
        edge_index = torch.tensor([[], []], dtype=torch.long) if not edges else torch.tensor(edges,
                                                                                             dtype=torch.long).t().contiguous()
        data = Data(x=torch.ones(len(ego), 1, dtype=torch.float32), edge_index=edge_index,
                    y=torch.tensor(item_y, dtype=torch.float32))

        return data
#%%
class Model(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, num_layers=1, dropout=0.1, batch_s=512, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.gin = GIN(in_channels, hidden_channels, num_layers, out_channels, dropout, train_eps=True)#normalization
        self.slp = MLP([1, out_channels])
        self.mlp = MLP([out_channels * 2, out_channels, out_channels, 1], dropout=dropout)

        # compute the difference between the two vectors y_hat and y as loss function
        self.loss = nn.L1Loss()
        self.batch_size = batch_s
        self.lr = lr

    def forward(self, x, edge_index, batch):
        index = torch.cat([torch.tensor([0], device=self.device), batch.long().bincount().cumsum(dim=0)])[:-1]
        x = self.gin(x, edge_index)
        x = x[index]
        x_sub = self.slp(torch.ones(x.size(0), 1, device=self.device))
        x = torch.cat([x, x_sub], dim=1)
        x = self.mlp(x).reshape(-1)
        x = torch.relu(x)
        # x = torch.round(x)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = self.loss(y_hat, data.y)
        self.log('train_loss', loss, on_epoch=False, batch_size=self.batch_size, prog_bar=True, logger=True)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = self.loss(y_hat, data.y)
        self.log('val_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True, logger=True)

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        # y_hat = torch.round(y_hat)
        loss = self.loss(y_hat, data.y)
        print('y_hat: ')
        print(y_hat)
        print('y: ')
        print(data.y)
        self.log('test_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True, logger=True)

    def predict_step(self, data, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def backward(self, loss, *args, **kwargs):
        loss.backward()
#%%
if __name__ == '__main__':
    # visualize the model
    # model = Model(in_channels=1, hidden_channels=64, out_channels=32, num_layers=1, dropout=0.1, batch_s=2, lr=0.01)
    # x = torch.ones(2, 1)
    # edge_index = torch.tensor([[0, 1],
    #                            [0, 1]], dtype=torch.long)
    # batch = torch.tensor([0, 1], dtype=torch.long)
    # dot = make_dot(model(x, edge_index, batch), params=dict(model.named_parameters()))
    # dot.view()
    torch.set_float32_matmul_precision('medium')
    # list of datasets want to use
    dataset = ['ENZYMES']
    # list of query graphs
    query_graph = [nx.graph_atlas(14)]
    # for qg in query_graph:
    #     nx.draw(qg, with_labels=True, font_weight='bold')
    batch_size = 512
    datamodule = ENDataModule(dataset, query_graph, hop=nx.diameter(query_graph[0]), batch_s=batch_size, num_workers=8)
    model = Model(in_channels=1, hidden_channels=64, out_channels=32, num_layers=3, dropout=0.1, batch_s=batch_size, lr=0.01)
    logger = loggers.TensorBoardLogger('./', version=0)
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', num_sanity_val_steps=0, logger=logger, log_every_n_steps=10, callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
    trainer.fit(model, datamodule)
    trainer.test(ckpt_path='last', datamodule=datamodule)
#%%
# display to tensorboard and show in the jupyter notebook
%reload_ext tensorboard
%tensorboard --logdir lightning_logs/
#%%
# visualize the model
model = Model(in_channels=1, hidden_channels=64, out_channels=32, num_layers=1, dropout=0.1, batch_s=2, lr=0.01)
x = torch.ones(2, 1)
edge_index = torch.tensor([[0, 1],
                           [0, 1]], dtype=torch.long)
batch = torch.tensor([0, 1], dtype=torch.long)
dot = make_dot(model(x, edge_index, batch), params=dict(model.named_parameters()))
dot.view()