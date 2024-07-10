import torch
import pickle
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url, Dataset
from torch_geometric.utils import from_networkx
import shutil, os, json, sys

from graph_datasets.SyntheticDatasetGenerator import SyntheticDatasetGenerator
import graph_datasets
print(graph_datasets.__file__)


with open(os.path.join(os.path.dirname(graph_datasets.__file__),"../../../src/graph_datasets","config", "dataset_testing.json")) as f:
    synteticdataset_settings = json.load(f)
# reasoning_package_path = ament_index_python.get_package_share_directory("graph_reasoning")
with open(os.path.join(os.path.dirname(graph_datasets.__file__),"../../../src/graph_reasoning","config", "pard_training.json")) as f:
    graph_reasoning_settings = json.load(f)

class SceneGraphs(InMemoryDataset):
    def __init__(self, dataset_name, root='data/ARM', split='train', transform=None, pre_transform=None):
        self.name = dataset_name
        self.split = split
        super().__init__(f'{root}/{dataset_name}', transform, pre_transform)
        self.data, self.slices = self.create_dataset()
        # print(f'dbg self.data {self.data}')
        # print(f'dbg self.data.x {self.data.x}')
        # print(f'dbg self.data.edge_index {self.data.edge_index}')
        # print(f'dbg self.data.edge_attr {self.data.edge_attr}')
        # print(f'dbg self.data.idx {self.data.idx}')
        # print(f'dbg self.data.y {self.data.y}')
        # print(f'dbg self.slices {self.slices}')
        # print(f'dbg self.slices {self.slices}')
        # print(f'dbg self.slices [x] {self.slices["x"].shape, self.slices["x"]}')
        # print(f'dbg self.slices [edge_index] {self.slices["edge_index"].shape, self.slices["edge_index"]}')
        # print(f'dbg self.slices [edge_attr] {self.slices["edge_attr"].shape, self.slices["edge_attr"]}')
        # print(f'dbg self.slices [y] {self.slices["y"].shape, self.slices["y"]}')
        # print(f'dbg self.slices [idx] {self.slices["idx"].shape, self.slices["idx"]}')
        # print(f'dbg self.len {self.len()}')
        # kajsdfh

    @property
    def raw_file_names(self):
        return f"{self.name}.pkl"
    
    @property
    def processed_file_names(self):
        return f'{self.name}_{self.split}.pt' 
    
    def download(self):
        pass
        # if self.name == 'ENZYMES':
        #     print(f'Downloading ENZYMES dataset ... to {self.raw_paths[0]}')
        #     src = ARMDataset.enyzme_url
        #     download_url(src, self.raw_dir)
        #     # TODO: the format is not correct, GDSS dataset is not aligning with ARM. 
        # else:
        #     print('Copying files from source path ...')
        #     src = os.path.join(self.root, os.pardir, f"{self.name}.pkl")
        #     shutil.copyfile(src, self.raw_paths[0])

        # merged_graph, slices = self.create_dataset()
        # print(merged_graph)
        # adxfg

    def process(self):
        pass
        # # Read data into huge `Data` list. 
        # graphs = pickle.load(open(self.raw_paths[0], 'rb'))
        # # transform graphs from nx to data list 
        # data_list = [from_networkx(nx_graphs) for nx_graphs in graphs]
        # if hasattr(data_list[0], 'label'):
        #     labels = [item for data in data_list for item in data.label ]
        #     node_encoding = {string: i for i, string in enumerate(set(labels))}
        #     edge_labels = [item for data in data_list for item in data.edge_label ]
        #     edge_encoding = {string: i for i, string in enumerate(set(edge_labels))}
        #     def _transform(data):
        #         data.x = torch.tensor([node_encoding[label] for label in data.label])
        #         data.edge_attr = torch.tensor([edge_encoding[label] for label in data.edge_label])
        #         del data.label, data.edge_label
        #         return data
        #     data_list = [_transform(data) for data in data_list]

        
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # num_graphs = len(data_list)
        # num_train = int(num_graphs * 0.8)
        # num_val = int(num_train * 0.2)
        # num_train = num_train - num_val
        
        # lists = {
        #     'train': data_list[num_val:num_val+num_train],
        #     'val': data_list[:num_val],
        #     'test': data_list[num_train+num_val:]
        # }
        # for split in ['train', 'val', 'test']:
        #     torch.save(self.collate(lists[split]), osp.join(self.processed_dir, f'{self.name}_{split}.pt'))

    def create_dataset(self):
        dataset_generator = SyntheticDatasetGenerator(synteticdataset_settings, logger = None, report_path = None, dataset_name = "test")
        dataset_generator.create_dataset()
        settings_hdata = graph_reasoning_settings["hdata"]
        filtered_nxdataset = dataset_generator.get_filtered_datset(settings_hdata["nodes"],settings_hdata["edges"])["noise"]
        merged_graph, slices = dataset_generator.merge_graphs_type_as_x(filtered_nxdataset)
        return merged_graph, slices