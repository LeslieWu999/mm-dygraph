import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

from utils.utils import NeighborSampler
from models.modules import TimeEncoder, MergeLayer, MultiHeadAttention

from .moe import MoE
from .ccffcc import cfcbundle


class Liquid(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, structure_dim:int = 64,model_name: str = 'Liquid', num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 src_node_mean_time_shift: float = 0.0, src_node_std_time_shift: float = 1.0, dst_node_mean_time_shift_dst: float = 0.0,
                 dst_node_std_time_shift: float = 1.0, max_input_sequence_length: int = 512, device: str = 'cpu',fusion_method : str ='IB'):
        """
        General framework for memory-based models, support TGN, DyRep and JODIE.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        """
        super(Liquid, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift
        self.structure_feat_dim = 5
        # self.structure_feat_dim = 3
        self.max_input_sequence_length = max_input_sequence_length
        self.structure_memory_dim = 64

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_sampler = neighbor_sampler

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregator()

        # memory modules
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        self.structure_encoder=nn.Sequential(
            nn.Linear(in_features=self.structure_feat_dim, out_features=self.node_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim),
            nn.Dropout(0.1)
            )
        self.memory_updater = SalimMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)

        # embedding module
        self.embedding_module = GraphAttentionEmbedding(node_raw_features=self.node_raw_features,
                                                            edge_raw_features=self.edge_raw_features,
                                                            neighbor_sampler=neighbor_sampler,
                                                            time_encoder=self.time_encoder,
                                                            node_feat_dim=self.node_feat_dim,
                                                            edge_feat_dim=self.edge_feat_dim,
                                                            time_feat_dim=self.time_feat_dim,
                                                            num_layers=self.num_layers,
                                                            num_heads=self.num_heads,
                                                            dropout=self.dropout)

        # self.proj=nn.Linear(self.memory_dim, self.node_feat_dim)

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.memory_dim*2, out_features=self.node_feat_dim),
            nn.Dropout(0.1)
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.node_feat_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.node_feat_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.node_feat_dim, bias=True),
            'structure': nn.Linear(in_features=self.structure_feat_dim, out_features=self.node_feat_dim, bias=True),
        })

        self.num_channels = 4

        self.fusion_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(self.num_channels * self.node_feat_dim, self.num_channels * self.node_feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.num_channels * self.node_feat_dim))
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=(self.num_channels) * self.node_feat_dim, out_features=self.node_feat_dim, bias=True)

        # self.proj=nn.Linear(self.node_feat_dim//4+1, self.node_feat_dim)

        self.tem_norm=nn.LayerNorm(self.node_feat_dim)
        self.struc_norm=nn.LayerNorm(self.node_feat_dim)

        if fusion_method=='IB':
            self.mixer=VIBModel(self.node_feat_dim,self.node_feat_dim,self.node_feat_dim,beta=1e-3)
        elif fusion_method=='cat':
            self.mixer=catcat()
        elif fusion_method=='add':
            self.mixer=addadd()

        self.position_feat_dim = 172
        self.walk_length = 2

        # self.position_encoder = PositionEncoder(position_feat_dim=self.position_feat_dim, walk_length=self.walk_length, device=device)

        # self.walk_encoder = BiLSTMEncoder(input_dim=self.position_feat_dim, hidden_dim=self.position_feat_dim)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray, edges_are_positive: bool = True, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param edges_are_positive: boolean, whether the edges are positive,
        determine whether to update the memories and raw messages for nodes in src_node_ids and dst_node_ids or not
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        # we need to use self.get_updated_memory instead of self.update_memory based on positive_node_ids directly,
        # because the graph attention embedding module in TGN needs to additionally access memory of neighbors.
        # so we return all nodes' memory with shape (num_nodes, ) by using self.get_updated_memory
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )

        uptime = 0

        tup1 = time.time()

        updated_node_memories, updated_node_last_updated_times,exl,t1,t2,t3 = self.get_updated_memories(node_ids=np.array(range(self.num_nodes)),
                                                                                           node_raw_messages=self.memory_bank.node_raw_messages)
        # updated_node_memories, updated_node_last_updated_times,exl = self.get_updated_memories(node_ids=np.array(range(self.num_nodes)),
        # #                                                                                    node_raw_messages=self.memory_bank.node_raw_messages)
        # updated_node_memories, updated_node_last_updated_times = self.get_updated_memories(node_ids=np.array(range(self.num_nodes)),
        #                                                                                    node_raw_messages=self.memory_bank.node_raw_messages)

        tup2 = time.time()

        uptime += tup2 - tup1

        # compute the node temporal embeddings using the embedding module

            # Tensor, shape (2 * batch_size, node_feat_dim)
        node_embeddings = self.embedding_module.compute_node_temporal_embeddings(node_memories=updated_node_memories,
                                                                                    node_ids=node_ids,
                                                                                    node_interact_times=np.concatenate([node_interact_times,
                                                                                                                        node_interact_times]),
                                                                                    current_layer_num=self.num_layers,
                                                                                    num_neighbors=num_neighbors)
        # two Tensors, with shape (batch_size, node_feat_dim)
        src_node_embeddings, dst_node_embeddings = node_embeddings[:len(src_node_ids)], node_embeddings[len(src_node_ids): len(src_node_ids) + len(dst_node_ids)]
        # gongxainlinjv
        src_structure_embedding,dst_structure_embedding=self.compute_coocurrence(src_node_ids, dst_node_ids, node_interact_times,positive=edges_are_positive)
        # #randomwalks
        # src_node_multi_hop_graphs = self.neighbor_sampler.get_multi_hop_neighbors(num_hops=self.walk_length, node_ids=src_node_ids,
        #                                                                           node_interact_times=node_interact_times, num_neighbors=num_neighbors)
        # # tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        # dst_node_multi_hop_graphs = self.neighbor_sampler.get_multi_hop_neighbors(num_hops=self.walk_length, node_ids=dst_node_ids,
        #                                                                           node_interact_times=node_interact_times, num_neighbors=num_neighbors)

        # count the appearances appearances of nodes in the multi-hop graphs that are generated by random walks that
        # start from src node in src_node_ids and dst node in dst_node_ids
        # self.position_encoder.count_nodes_appearances(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
        #                                               node_interact_times=node_interact_times,
        #                                               src_node_multi_hop_graphs=src_node_multi_hop_graphs,
        #                                               dst_node_multi_hop_graphs=dst_node_multi_hop_graphs)

        # src_structure_embedding = self.compute_node_temporal_embeddings(node_ids=src_node_ids, node_interact_times=node_interact_times,
        #                                                             node_multi_hop_graphs=src_node_multi_hop_graphs, num_neighbors=num_neighbors)
        # # Tensor, shape (batch_size, node_feat_dim)
        # dst_structure_embedding = self.compute_node_temporal_embeddings(node_ids=dst_node_ids, node_interact_times=node_interact_times,
        #                                                             node_multi_hop_graphs=dst_node_multi_hop_graphs, num_neighbors=num_neighbors)


        if exl is None:
            exl=0
            t1,t2,t3 = 0,0,0

        if edges_are_positive:
            assert edge_ids is not None
            # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)

            upt3 = time.time()

            ext,tt1,tt2,tt3 = self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)
            # ext = self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)
            # self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)

            upt4 = time.time()
            if ext is None:
                ext=0
                tt1,tt2,tt3 = 0,0,0

            uptime += upt4 - upt3

            exl=exl+ext
            t1+=tt1
            t2+=tt2
            t3+=tt3

            # clear raw messages for source and destination nodes since we have already updated the memory using them
            self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

            # compute new raw messages for source and destination nodes
            unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=src_node_ids,
                                                                                                dst_node_ids=dst_node_ids,
                                                                                                dst_node_embeddings=dst_node_embeddings,
                                                                                                node_interact_times=node_interact_times,
                                                                                                edge_ids=edge_ids)
            unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=dst_node_ids,
                                                                                                dst_node_ids=src_node_ids,
                                                                                                dst_node_embeddings=src_node_embeddings,
                                                                                                node_interact_times=node_interact_times,
                                                                                                edge_ids=edge_ids)

            # store new raw messages for source and destination nodes
            self.memory_bank.store_node_raw_messages(node_ids=unique_src_node_ids, new_node_raw_messages=new_src_node_raw_messages)
            self.memory_bank.store_node_raw_messages(node_ids=unique_dst_node_ids, new_node_raw_messages=new_dst_node_raw_messages)

        # src_node_embeddings = self.tem_norm(src_node_embeddings)
        # dst_node_embeddings = self.tem_norm(dst_node_embeddings)

        # src_structure_embedding = self.struc_norm(src_structure_embedding)
        # dst_structure_embedding = self.struc_norm(dst_structure_embedding)

        # src_embeddings = torch.concat((src_node_embeddings, src_structure_embedding), dim=-1)
        # dst_embeddings = torch.concat((dst_node_embeddings, dst_structure_embedding), dim=-1)

        # src_embeddings = self.projection(src_embeddings)
        # dst_embeddings = self.projection(dst_embeddings)

        src_embeddings,exloss1 = self.mixer(src_node_embeddings,src_structure_embedding)
        dst_embeddings,exloss2 = self.mixer(dst_node_embeddings,dst_structure_embedding)

        exloss=exloss1+exloss2+exl
        # exloss=exloss1+exloss2

        # exloss=exl

        # src_embeddings=src_node_embeddings
        # dst_embeddings=dst_node_embeddings


        # src_node_embeddings = self.projection(torch.cat([src_node_embeddings,src_structure_embedding],dim=-1))
        # dst_node_embeddings = self.projection(torch.cat([dst_node_embeddings,dst_structure_embedding],dim=-1))

        # src_embeddings=src_node_embeddings
        # dst_embeddings=dst_node_embeddings

        # exloss=0

        return src_embeddings, dst_embeddings, exloss
    
    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, node_interact_times: np.ndarray, node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        given node interaction time node_interact_times and node multi-hop graphs node_multi_hop_graphs,
        return the temporal embeddings of nodes
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_multi_hop_graphs: tuple of three ndarrays, each array with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        :return:
        """
        # three ndarrays, each array with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = \
            self.convert_format_from_tree_to_array(node_ids=node_ids, node_interact_times=node_interact_times,
                                                   node_multi_hop_graphs=node_multi_hop_graphs, num_neighbors=num_neighbors)

        # ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        walks_valid_lengths = (nodes_neighbor_ids != 0).sum(axis=-1)

        # get time features of nodes in the multi-hop graphs
        # check that the time of start node in each walk should be identical to the node in the batch
        assert (nodes_neighbor_times[:, :, 0] == node_interact_times.repeat(repeats=num_neighbors ** self.walk_length, axis=0).
                reshape(len(node_interact_times), num_neighbors ** self.walk_length)).all()

        # get edge features of nodes in the multi-hop graphs
        # ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        # check that the edge ids of the target node is denoted by zeros
        assert (nodes_edge_ids[:, :, 0] == 0).all()


        # get position features of nodes in the multi-hop graphs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        neighbor_position_features = self.position_encoder(nodes_neighbor_ids=nodes_neighbor_ids)

        # encode the random walks by walk encoder
        # Tensor, shape (batch_size, self.output_dim)
        final_node_embeddings = self.walk_encoder(inputs=neighbor_position_features, lengths=walks_valid_lengths).mean(dim = -2)
        return final_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                        nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
            """
            pad the sequences for nodes in node_ids
            :param node_ids: ndarray, shape (batch_size, )
            :param node_interact_times: ndarray, shape (batch_size, )
            :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
            :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
            :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
            :param patch_size: int, patch size
            :param max_input_sequence_length: int, maximal number of neighbors for each node
            :return:
            """
            assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
            max_seq_length = 0
            # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
            for idx in range(len(nodes_neighbor_ids_list)):
                assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
                if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                    # cut the sequence by taking the most recent max_input_sequence_length interactions
                    nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                    nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                    nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
                if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                    max_seq_length = len(nodes_neighbor_ids_list[idx])

            # include the target node itself
            max_seq_length += 1
            if max_seq_length % patch_size != 0:
                max_seq_length += (patch_size - max_seq_length % patch_size)
            assert max_seq_length % patch_size  == 0

            # pad the sequences
            # three ndarrays with shape (batch_size, max_seq_length)
            padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.int32)
            padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.int32)
            padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

            for idx in range(len(node_ids)):
                padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
                padded_nodes_edge_ids[idx, 0] = 0
                padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

                if len(nodes_neighbor_ids_list[idx]) > 0:
                    padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                    padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                    padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

            # three ndarrays with shape (batch_size, max_seq_length)
            return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times
    def compute_coocurrence(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,positive:bool=True):
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        # pad the sequences of first-hop neighbors for source and destination nodes
        # src_padded_nodes_neighbor_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_edge_ids, ndarray, shape (batch_size, src_max_seq_length)
        # src_padded_nodes_neighbor_times, ndarray, shape (batch_size, src_max_seq_length)
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # dst_padded_nodes_neighbor_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_edge_ids, ndarray, shape (batch_size, dst_max_seq_length)
        # dst_padded_nodes_neighbor_times, ndarray, shape (batch_size, dst_max_seq_length)
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               max_input_sequence_length=self.max_input_sequence_length)

        # read src and dst neighbor memory (batch_size, max_seq_length, memory_dim)
        src_memory = self.memory_bank.get_struc_memories(src_node_ids).unsqueeze(1)
        dst_memory = self.memory_bank.get_struc_memories(dst_node_ids).unsqueeze(1)

        src_memory_short, src_memory_long = src_memory[:, :, :self.structure_memory_dim // 4], src_memory[:, :,self.structure_memory_dim // 4:]
        dst_memory_short, dst_memory_long = dst_memory[:, :, :self.structure_memory_dim // 4], dst_memory[:, :,self.structure_memory_dim // 4:]


        src_neighbor_memory = self.memory_bank.get_struc_memories(src_padded_nodes_neighbor_ids)
        dst_neighbor_memory = self.memory_bank.get_struc_memories(dst_padded_nodes_neighbor_ids)

        src_neighbor_memory_short, src_neighbor_memory_long = src_neighbor_memory[:, :,:self.structure_memory_dim // 4], src_neighbor_memory[:, :,self.structure_memory_dim // 4:]
        dst_neighbor_memory_short, dst_neighbor_memory_long = dst_neighbor_memory[:, :,:self.structure_memory_dim // 4], dst_neighbor_memory[:, :,self.structure_memory_dim // 4:]

        # compute co neighbor encoding (batch_size, max_seq_length, 1)
        pos_feature_src_src = torch.sum((src_memory_long.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_long) * (src_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_src_dst = torch.sum((dst_memory_long.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_long) * (src_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)

        pos_feature_dst_dst = torch.sum((dst_memory_long.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_long) * (dst_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_dst_src = torch.sum((src_memory_long.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_long) * (dst_neighbor_memory_long != 0).float(), dim=-1).unsqueeze(-1)

        pos_feature_src_src_short = torch.sum((src_memory_short.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_short) * (src_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_src_dst_short = torch.sum((dst_memory_short.repeat(1, src_padded_nodes_neighbor_ids.shape[1], 1) == src_neighbor_memory_short) * (src_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)

        pos_feature_dst_dst_short = torch.sum((dst_memory_short.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_short) * (dst_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)
        pos_feature_dst_src_short = torch.sum((src_memory_short.repeat(1, dst_padded_nodes_neighbor_ids.shape[1], 1) == dst_neighbor_memory_short) * (dst_neighbor_memory_short != 0).float(), dim=-1).unsqueeze(-1)

        src_coocur = ((torch.from_numpy(dst_node_ids).unsqueeze(1).repeat(1, src_padded_nodes_neighbor_ids.shape[
            1])) == torch.from_numpy(src_padded_nodes_neighbor_ids)).float().to(self.device).unsqueeze(-1)
        dst_coocur = ((torch.from_numpy(src_node_ids).unsqueeze(1).repeat(1, dst_padded_nodes_neighbor_ids.shape[
            1])) == torch.from_numpy(dst_padded_nodes_neighbor_ids)).float().to(self.device).unsqueeze(-1)
        
        src_padded_nodes_neighbor_structure_features = torch.cat([pos_feature_src_src, pos_feature_src_dst, pos_feature_src_src_short, pos_feature_src_dst_short, src_coocur], dim=-1)
        dst_padded_nodes_neighbor_structure_features = torch.cat([pos_feature_dst_dst, pos_feature_dst_src, pos_feature_dst_dst_short, pos_feature_dst_src_short, dst_coocur], dim=-1)

        
        # src_padded_nodes_neighbor_structure_features = torch.cat([pos_feature_src_src, pos_feature_src_dst, src_coocur], dim=-1)
        # dst_padded_nodes_neighbor_structure_features = torch.cat([pos_feature_dst_dst, pos_feature_dst_src, dst_coocur], dim=-1)

        src_padded_nodes_neighbor_structure_features = self.structure_encoder(src_padded_nodes_neighbor_structure_features).mean(dim=1)
        dst_padded_nodes_neighbor_structure_features = self.structure_encoder(dst_padded_nodes_neighbor_structure_features).mean(dim=1)

        
        # # get the features of the sequence of source and destination nodes
        # # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # # src_padded_nodes_edge_raw_features, Tensor, shape (batch_size, src_max_seq_length, edge_feat_dim)
        # # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        # src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
        #     self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
        #                       padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # # dst_padded_nodes_edge_raw_features, Tensor, shape (batch_size, dst_max_seq_length, edge_feat_dim)
        # # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        # dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
        #     self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
        #                       padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # # align the patch encoding dimension
        # # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        # src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_padded_nodes_neighbor_node_raw_features)
        # src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_padded_nodes_edge_raw_features)
        # src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_padded_nodes_neighbor_time_features)
        # src_patches_nodes_neighbor_structure_features = self.projection_layer['structure'](src_padded_nodes_neighbor_structure_features)

        # # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        # dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_padded_nodes_neighbor_node_raw_features)
        # dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_padded_nodes_edge_raw_features)
        # dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_padded_nodes_neighbor_time_features)
        # dst_patches_nodes_neighbor_structure_features = self.projection_layer['structure'](dst_padded_nodes_neighbor_structure_features)

        # batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        # src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        # dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        # patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        # patches_nodes_edge_raw_features = torch.cat([src_patches_nodes_edge_raw_features, dst_patches_nodes_edge_raw_features], dim=1)
        # patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        # patches_nodes_neighbor_structure_features = torch.cat([src_patches_nodes_neighbor_structure_features, dst_patches_nodes_neighbor_structure_features], dim=1)

        # patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features,
        #                 patches_nodes_neighbor_time_features, patches_nodes_neighbor_structure_features]
        # # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        # patches_data = torch.stack(patches_data, dim=2)
        # # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        # patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.node_feat_dim)

        # # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        # for fusion_layer in self.fusion_layer:
        #     patches_data = fusion_layer(patches_data)

        # # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        # src_patches_data = patches_data[:, : src_num_patches, :]
        # # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        # dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        # src_patches_data = torch.mean(src_patches_data, dim=1)
        # # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        # dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # # Tensor, shape (batch_size, node_feat_dim)
        # src_node_embeddings = self.output_layer(torch.cat([src_patches_data],dim=-1))
        # # Tensor, shape (batch_size, node_feat_dim)
        # dst_node_embeddings = self.output_layer(torch.cat([dst_patches_data],dim=-1))

        
        if positive:
            self.memory_bank.set_struc_memories(src_node_ids.flatten(), dst_padded_nodes_neighbor_ids)
            self.memory_bank.set_struc_memories(dst_node_ids.flatten(), src_padded_nodes_neighbor_ids)

            self.memory_bank.set_struc_memories(src_padded_nodes_neighbor_ids[:,0:].flatten(), np.expand_dims(dst_node_ids, 1).repeat(src_padded_nodes_neighbor_ids.shape[1],0))
            self.memory_bank.set_struc_memories(dst_padded_nodes_neighbor_ids[:,0:].flatten(), np.expand_dims(src_node_ids, 1).repeat(dst_padded_nodes_neighbor_ids.shape[1],0))

        return src_padded_nodes_neighbor_structure_features, dst_padded_nodes_neighbor_structure_features

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids).long()]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids).long()]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features


    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        get the updated memories based on node_ids and node_raw_messages (just for computation), but not update the memories
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)
        # get updated memory for all nodes with messages stored in previous batches (just for computation)
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        updated_node_memories, updated_node_last_updated_times,exl,t1,t2,t3 = self.memory_updater.get_updated_memories(unique_node_ids=unique_node_ids,
                                                                                                          unique_node_messages=unique_node_messages,
                                                                                                          unique_node_timestamps=unique_node_timestamps)
        # updated_node_memories, updated_node_last_updated_times,exl = self.memory_updater.get_updated_memories(unique_node_ids=unique_node_ids,
        #                                                                                             unique_node_messages=unique_node_messages,
        #                                                                                             unique_node_timestamps=unique_node_timestamps)
        # updated_node_memories, updated_node_last_updated_times = self.memory_updater.get_updated_memories(unique_node_ids=unique_node_ids,
        #                                                                                     unique_node_messages=unique_node_messages,
        #                                                                                     unique_node_timestamps=unique_node_timestamps)

        return updated_node_memories, updated_node_last_updated_times,exl,t1,t2,t3

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        update memories for nodes in node_ids
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)
        exl,t1,t2,t3 = 0,0,0,0
        # update the memories with the aggregated messages
        try:
            exl,t1,t2,t3 = self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                            unique_node_timestamps=unique_node_timestamps)
        except:
            # exl = self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
            #                                 unique_node_timestamps=unique_node_timestamps)
            self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                    unique_node_timestamps=unique_node_timestamps)
            
        # self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
        #                             unique_node_timestamps=unique_node_timestamps)

        return exl,t1,t2,t3

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, dst_node_embeddings: torch.Tensor,
                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        compute new raw messages for nodes in src_node_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param dst_node_embeddings: Tensor, shape (batch_size, node_feat_dim)
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        # For DyRep, use destination_node_embedding aggregated by graph attention module for message encoding
        if self.model_name == 'DyRep':
            dst_node_memories = dst_node_embeddings
        else:
            dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = torch.from_numpy(node_interact_times).float().to(self.device) - \
                               self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(len(src_node_ids), -1)

        # Tensor, shape (batch_size, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat([src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        assert self.model_name in ['Liquid'], f'Neighbor sampler is not defined in model {self.model_name}!'
        self.embedding_module.neighbor_sampler = neighbor_sampler
        self.neighbor_sampler = neighbor_sampler
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.embedding_module.neighbor_sampler.seed is not None
            self.embedding_module.neighbor_sampler.reset_random_state()
            self.neighbor_sampler.reset_random_state()
    def convert_format_from_tree_to_array(self, node_ids: np.ndarray, node_interact_times: np.ndarray, node_multi_hop_graphs: tuple, num_neighbors: int = 20):
        """
        convert the multi-hop graphs from tree-like data format to aligned array-like format
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = node_multi_hop_graphs

        # add the target node to the list to generate random walks in array-like format
        nodes_neighbor_ids = [node_ids[:, np.newaxis]] + nodes_neighbor_ids
        # follow the CAWN official implementation, the edge ids of the target node is denoted by zeros
        nodes_edge_ids = [np.zeros((len(node_ids), 1)).astype(np.longlong)] + nodes_edge_ids
        nodes_neighbor_times = [node_interact_times[:, np.newaxis]] + nodes_neighbor_times

        array_format_data_list = []
        for tree_format_data in [nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times]:
            # num_last_hop_neighbors equals to num_neighbors ** self.walk_length
            batch_size, num_last_hop_neighbors, walk_length_plus_1, dtype = \
                tree_format_data[0].shape[0], tree_format_data[-1].shape[-1], len(tree_format_data), tree_format_data[0].dtype
            assert batch_size == len(node_ids) and num_last_hop_neighbors == num_neighbors ** self.walk_length and walk_length_plus_1 == self.walk_length + 1
            # record the information of random walks with num_last_hop_neighbors paths, where each path has length walk_length_plus_1 (include the target node)
            # ndarray, shape (batch_size, num_last_hop_neighbors, walk_length_plus_1)
            array_format_data = np.empty((batch_size, num_last_hop_neighbors, walk_length_plus_1), dtype=dtype)
            for hop_idx, hop_data in enumerate(tree_format_data):
                assert (num_last_hop_neighbors % hop_data.shape[-1] == 0)
                # pad the data at each hop to be the same shape with the last hop data (which has the most number of neighbors)
                # repeat the traversed nodes in tree_format_data to get the aligned array-like format
                array_format_data[:, :, hop_idx] = np.repeat(hop_data, repeats=num_last_hop_neighbors // hop_data.shape[-1], axis=1)
            array_format_data_list.append(array_format_data)
        # three ndarrays with shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        return array_format_data_list[0], array_format_data_list[1], array_format_data_list[2]


# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []

        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        # ndarray, shape (num_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.stru_memory_dim = 64

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)

        self.struc_memories = nn.Parameter(torch.zeros((self.num_nodes, self.stru_memory_dim+self.stru_memory_dim//4), dtype=torch.long), requires_grad=False)
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.struc_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_num_nodes(self):
        """
        get all node ids in the memory bank
        :return: ndarray, shape (num_nodes, ), array of node ids
        """
        return self.num_nodes

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories


    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.struc_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data, self.struc_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[1].clone(), backup_memory_bank[2].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[3].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()
        self.struc_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]


    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])

    def get_struc_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.struc_memories[torch.from_numpy(node_ids).to(torch.long)].long()
    def set_struc_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        node_ids = torch.from_numpy(node_ids).unsqueeze(1).repeat(1, updated_node_memories.shape[1]).to(torch.long)

        hash_short = self.hash_value_short(updated_node_memories).astype(np.int32)
        # hash_short2 = self.hash_value_short(updated_node_memories,2).astype(np.int)
        hash_long = self.hash_value_long(updated_node_memories).astype(np.int32)
        # hash_long2 = self.hash_value_long(updated_node_memories,3).astype(np.int)
        updated_node_memories = torch.from_numpy(updated_node_memories).to(torch.long).to(self.struc_memories.device)

        self.struc_memories[node_ids, hash_short] = updated_node_memories
        self.struc_memories[node_ids, hash_long] = updated_node_memories

    def hash_array(self, node_ids, seed):
        return (node_ids * (seed % 100) + node_ids^3 * ((seed % 100) + 1) + node_ids^5 * ((seed % 100) + 3))

    def hash_value_short(self, node_ids, seed=1):
        return self.hash_array(node_ids, seed)%(self.stru_memory_dim//4)
    def hash_value_long(self, node_ids, seed=2):
        return self.hash_array(node_ids, seed)%self.stru_memory_dim+self.stru_memory_dim//4



class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        """
        update memories for nodes in unique_node_ids
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, return without updating operations
        if len(unique_node_ids) <= 0:
            return 0, 0, 0, 0

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        exl,t1,t2,t3 = 0 ,0 ,0 ,0

        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)

        updated_node_memories,exl,t1,t2,t3 = self.memory_updater(unique_node_messages, node_memories)

        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        return exl,t1,t2,t3



        # secure_node_memorys = self.memory_bank.get_substitute_memories(secure_nodes)[0]
        # equal_node_memorys = self.memory_bank.get_substitute_memories(equal_nodes)[1]



        # # Tensor, shape (num_unique_node_ids, memory_dim)
        # node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        # # Tensor, shape (num_unique_node_ids, memory_dim)
        # updated_node_memories,exl,t1,t2,t3 = self.memory_updater(unique_node_messages, node_memories)

        # last_updated_times = self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids)
        
        
        # updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        # # update memories for nodes in unique_node_ids
        # self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        # # update last updated times for nodes in unique_node_ids
        # self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        # return exl, t1, t2, t3

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        """
        get updated memories based on unique_node_ids, unique_node_messages and unique_node_timestamps
        (just for computation), but not update the memories
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, directly return node_memories and node_last_updated_times without updating
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone(), 0, 0, 0, 0

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        updated_node_memories = self.memory_bank.node_memories.data.clone()

        updated_node_memories[torch.from_numpy(unique_node_ids)],exl,t1,t2,t3 = self.memory_updater(unique_node_messages,updated_node_memories[torch.from_numpy(unique_node_ids)])

        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        if exl is None:
            exl=0

        return updated_node_memories, updated_node_last_updated_times,exl,t1,t2,t3


class GRUMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(GRUMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class RNNMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        RNN-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(RNNMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)

class CFCMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(CFCMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = cfcbundle(input_size=message_dim, hidden_size=memory_dim)

class SalimMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(SalimMemoryUpdater, self).__init__(memory_bank)

        print(message_dim,memory_dim)

        self.memory_updater = MoE(message_dim, memory_dim, 8, memory_dim, k=3, noisy_gating=True)

class catcell(nn.Module):

    def __init__(self, message_dim: int, memory_dim: int):
        """
        CatCell, a simple linear layer for memory update.
        :param input_dim: int, dimension of input
        :param hidden_dim: int, dimension of hidden state
        """
        super(catcell, self).__init__()

        self.moecell = cfcbundle(message_dim, memory_dim)
        self.grucell = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)
        self.merge = nn.Linear(memory_dim+memory_dim,memory_dim)

    def forward(self, x,hx):
        """
        forward function
        :param x: Tensor, shape (batch_size, input_dim)
        :return: Tensor, shape (batch_size, hidden_dim)
        """
        out1 = self.moecell(x,hx)
        out2 = self.grucell(x,hx)
        return self.merge(torch.cat([out1,out2],dim=1)),0
class votecell(nn.Module):

    def __init__(self, message_dim: int, memory_dim: int):
        """
        VoteCell, a simple linear layer for memory update.
        :param input_dim: int, dimension of input
        :param hidden_dim: int, dimension of hidden state
        """
        super(votecell, self).__init__()

        self.moecell = cfcbundle(message_dim, memory_dim)
        self.grucell = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

    def forward(self, x,hx):
        """
        forward function
        :param x: Tensor, shape (batch_size, input_dim)
        :return: Tensor, shape (batch_size, hidden_dim)
        """
        out1 = self.moecell(x,hx)
        out2 = self.grucell(x,hx)
        return 0.5*out1+0.5*out2,0

class catMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(catMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = catcell(message_dim, memory_dim)
class VoteMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(VoteMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = votecell(message_dim, memory_dim)


# Embedding-related Modules
class TimeProjectionEmbedding(nn.Module):

    def __init__(self, memory_dim: int, dropout: float):
        """
        Time projection embedding module.
        :param memory_dim: int, dimension of node memories
        :param dropout: float, dropout rate
        """
        super(TimeProjectionEmbedding, self).__init__()

        self.memory_dim = memory_dim
        self.dropout = nn.Dropout(dropout)

        self.linear_layer = nn.Linear(1, self.memory_dim)

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray, node_time_intervals: torch.Tensor):
        """
        compute node temporal embeddings using the embedding projection operation in JODIE
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_time_intervals: Tensor, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        source_embeddings = self.dropout(node_memories[torch.from_numpy(node_ids)] * (1 + self.linear_layer(node_time_intervals.unsqueeze(dim=1))))

        return source_embeddings


class GraphAttentionEmbedding(nn.Module):

    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler,
                 time_encoder: TimeEncoder, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        """
        Graph attention embedding module.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_encoder: TimeEncoder
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(GraphAttentionEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        

        # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])
        # self.proj = nn.Linear(self.node_feat_dim, self.node_feat_dim//4+1)
    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray, node_interact_times: np.ndarray, 
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        """

        assert (current_layer_num >= 0)
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        # shape (batch_size, node_feat_dim)
        # add memory and node raw features to get node features
        # note that when using getting values of the ids from Tensor, convert the ndarray to tensor to avoid wrong retrieval
        node_features = node_memories[torch.from_numpy(node_ids)] + self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_features
        else:
            # get source node representations by aggregating embeddings from the previous (curr_layers - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                       node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_times ndarray, shape (batch_size, num_neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                                node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)

            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(node_ids.shape[0], num_neighbors, self.node_feat_dim)

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)

            return output


def compute_src_dst_node_time_shifts(src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
    """
    compute the mean and standard deviation of time shifts
    :param src_node_ids: ndarray, shape (*, )
    :param dst_node_ids:: ndarray, shape (*, )
    :param node_interact_times: ndarray, shape (*, )
    :return:
    """
    src_node_last_timestamps = dict()
    dst_node_last_timestamps = dict()
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []
    for k in range(len(src_node_ids)):
        src_node_id = src_node_ids[k]
        dst_node_id = dst_node_ids[k]
        node_interact_time = node_interact_times[k]
        if src_node_id not in src_node_last_timestamps.keys():
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps.keys():
            dst_node_last_timestamps[dst_node_id] = 0
        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])
        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time
    assert len(src_node_all_time_shifts) == len(src_node_ids)
    assert len(dst_node_all_time_shifts) == len(dst_node_ids)
    src_node_mean_time_shift = np.mean(src_node_all_time_shifts)
    src_node_std_time_shift = np.std(src_node_all_time_shifts)
    dst_node_mean_time_shift_dst = np.mean(dst_node_all_time_shifts)
    dst_node_std_time_shift = np.std(dst_node_all_time_shifts)

    return src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift

class VIBModel(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, beta=1e-3):

        super(VIBModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.beta = beta


        # self.encoder1 = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim),
        #     nn.ReLU(inplace=True),

        # )
        # self.fc_mu_1 = nn.Linear(latent_dim, latent_dim)
        # self.fc_std_1 = nn.Linear(latent_dim, latent_dim)

        # self.encoder2 = nn.Sequential(
        #     nn.Linear(input_dim, latent_dim),
        #     nn.ReLU(inplace=True),

        # )
        # self.fc_mu_2 = nn.Linear(latent_dim, latent_dim)
        # self.fc_std_2 = nn.Linear(latent_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim*2, output_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu=nn.Linear(output_dim, output_dim)
        self.fc_std=nn.Linear(output_dim, output_dim)

        self.weight_init()


    def weight_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def loss(self, mu, std):
        kl_loss = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        total_loss = self.beta * kl_loss
        return total_loss

    def forward(self, x1,x2):

        if x1.dim() > 2:
            x1 = x1.view(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.view(x2.size(0), -1)

        # x1=self.encoder1(x1)
        # mu1=self.fc_mu_1(x1)
        # std1=F.softplus(self.fc_std_1(x1)-5, beta=1)
        # z1=self.reparametrize(mu1,std1)

        # x2=self.encoder2(x2)
        # mu2=self.fc_mu_2(x2)
        # std2=F.softplus(self.fc_std_2(x2)-5, beta=1)
        # z2=self.reparametrize(mu2,std2)

        z=self.encoder(torch.cat([x1,x2],dim=1))
        muz=self.fc_mu(z)
        stdz=F.softplus(self.fc_std(z)-5, beta=1)
        z=self.reparametrize(muz,stdz)

        exloss=self.loss(muz,stdz)
        # exloss=self.loss(muz,stdz) 

        return  z,exloss

    def reparametrize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

class catcat(nn.Module):
    def __init__(self):
        super(catcat,self).__init__()
    def forward(self,x1,x2):
        return torch.cat([x1,x2],dim=1)
    
class addadd(nn.Module):
    def __init__(self):
        super(addadd,self).__init__()
    def forward(self,x1,x2):
        return x1+x2
    
class PositionEncoder(nn.Module):

    def __init__(self, position_feat_dim: int, walk_length: int, device: str = 'cpu'):
        """
        Position encoder that computes each node position features.
        :param position_feat_dim: int, dimension of position features (encodings)
        :param walk_length: int, length of each random walk
        :param device: str, device
        """
        super(PositionEncoder, self).__init__()
        self.position_feat_dim = position_feat_dim
        self.walk_length = walk_length
        self.device = device

        # two-layered feed forward network with ReLU activation
        self.position_encode_layer = nn.Sequential(nn.Linear(in_features=self.walk_length + 1, out_features=self.position_feat_dim),
                                                   nn.ReLU(),
                                                   nn.Linear(in_features=self.position_feat_dim, out_features=self.position_feat_dim))

    def count_nodes_appearances(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                src_node_multi_hop_graphs: tuple, dst_node_multi_hop_graphs: tuple):
        """
        count the appearances of nodes in the multi-hop graphs that are generated by random walks starting from src and dst nodes
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param src_node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :param dst_node_multi_hop_graphs: tuple, each element in the tuple is a list of self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        :return:
        """
        # use node id and interaction timestamp to identify a node in the multi-hop graph
        # src_nodes_neighbor_ids and src_nodes_neighbor_times are lists, each list contains self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        src_nodes_neighbor_ids, _, src_nodes_neighbor_times = src_node_multi_hop_graphs
        # dst_nodes_neighbor_ids and dst_nodes_neighbor_times are lists, each list contains self.walk_length ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        dst_nodes_neighbor_ids, _, dst_nodes_neighbor_times = dst_node_multi_hop_graphs

        # dictionary, {node_identity (key): ndarray with shape (2, self.walk_length + 1) (value)}
        # store the appearances of all the nodes in the multi-hop graphs that are generated by random walks starting from src and dst nodes
        self.nodes_appearances = {}
        # get the multi-hop information for each node
        for idx, (src_node_id, dst_node_id, node_interact_time) in enumerate(zip(src_node_ids, dst_node_ids, node_interact_times)):
            # src_node_neighbor_ids, list of ndarrays, each ndarray with shape (num_neighbors ** current_hop)
            src_node_neighbor_ids = [src_nodes_single_hop_neighbor_ids[idx] for src_nodes_single_hop_neighbor_ids in src_nodes_neighbor_ids]
            src_node_neighbor_times = [src_nodes_single_hop_neighbor_times[idx] for src_nodes_single_hop_neighbor_times in src_nodes_neighbor_times]
            dst_node_neighbor_ids = [dst_nodes_single_hop_neighbor_ids[idx] for dst_nodes_single_hop_neighbor_ids in dst_nodes_neighbor_ids]
            dst_node_neighbor_times = [dst_nodes_single_hop_neighbor_times[idx] for dst_nodes_single_hop_neighbor_times in dst_nodes_neighbor_times]

            # dictionary, {node_identity (key): ndarray with shape (2, self.walk_length + 1) (value)}
            # store the appearances of nodes in the multi-hop graphs that are generated by random walks starting from src_node_id and dst_node_id
            tmp_nodes_appearances = {}
            # add the information of src_node and dst_node to the lists
            src_node_neighbor_ids, src_node_neighbor_times = [[src_node_id]] + src_node_neighbor_ids, [[node_interact_time]] + src_node_neighbor_times
            dst_node_neighbor_ids, dst_node_neighbor_times = [[dst_node_id]] + dst_node_neighbor_ids, [[node_interact_time]] + dst_node_neighbor_times
            for current_hop in range(self.walk_length + 1):
                for src_node_neighbor_id, src_node_neighbor_time, dst_node_neighbor_id, dst_node_neighbor_time in \
                        zip(src_node_neighbor_ids[current_hop], src_node_neighbor_times[current_hop], dst_node_neighbor_ids[current_hop], dst_node_neighbor_times[current_hop]):

                    # follow the CAWN official implementation, use the batch index and node id to represent the node key
                    src_node_key = '-'.join([str(idx), str(src_node_neighbor_id)])
                    dst_node_key = '-'.join([str(idx), str(dst_node_neighbor_id)])

                    if src_node_key not in tmp_nodes_appearances:
                        # create a ndarray with shape (2, self.walk_length + 1) for the src node to record its appearances
                        tmp_nodes_appearances[src_node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
                    if dst_node_key not in tmp_nodes_appearances:
                        # create a ndarray with shape (2, self.walk_length + 1) for the dst node to record its appearances
                        tmp_nodes_appearances[dst_node_key] = np.zeros((2, self.walk_length + 1), dtype=np.float32)

                    # count the appearances of each node in the multi-hop graphs that are generated by random walks starting from src_node_id and dst_node_id
                    # for each node, tmp_nodes_appearances[node_key][0, :] records the node appearances in the random walks starting from src_node_id
                    # while tmp_nodes_appearances[node_key][1, :] records the node appearances in the random walks starting from dst_node_id
                    # number of neighbors at the current hop
                    num_current_hop_neighbors = len(src_node_neighbor_ids[current_hop])
                    # convert into landing probabilities by normalizing with k hop sampling number
                    tmp_nodes_appearances[src_node_key][0, current_hop] += 1 / num_current_hop_neighbors
                    tmp_nodes_appearances[dst_node_key][1, current_hop] += 1 / num_current_hop_neighbors
            # set the appearances of the padded node (with zero index) to zeros
            tmp_nodes_appearances['-'.join([str(idx), str(0)])] = np.zeros((2, self.walk_length + 1), dtype=np.float32)
            self.nodes_appearances.update(tmp_nodes_appearances)

    def forward(self, nodes_neighbor_ids: np.ndarray):
        """
        compute the position features of nodes in nodes_neighbor_ids
        :param nodes_neighbor_ids: ndarray, shape shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1)
        :return:
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        """
        # batch_indices -> array([[[0, ..., 0,], ..., [0, ..., 0,]], [[1, ..., 1], ..., [1, ..., 1]] ..., [[batch - 1, ..., batch - 1], ..., [batch - 1, ..., batch - 1]]])
        batch_indices = np.arange(nodes_neighbor_ids.shape[0]).repeat(nodes_neighbor_ids.shape[1] * nodes_neighbor_ids.shape[2]).reshape(nodes_neighbor_ids.shape)

        # list of string keys, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1))
        batch_keys = ['-'.join([str(batch_indices[i][j][k]), str(nodes_neighbor_ids[i][j][k])])
                      for i in range(batch_indices.shape[0]) for j in range(batch_indices.shape[1]) for k in range(batch_indices.shape[2])]

        # unique_keys, ndarray, shape (num_unique_keys, )
        # inverse_indices, ndarray, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1))
        # we can use unique_keys[inverse_indices] to reconstruct the original input
        unique_keys, inverse_indices = np.unique(batch_keys, return_inverse=True)
        # self.nodes_appearances, dictionary, {node_identity (key): ndarray with shape (2, self.walk_length + 1) (value)}
        # unique_node_appearances, ndarray, shape (num_unique_keys, 2, self.walk_length + 1)
        unique_node_appearances = np.array([self.nodes_appearances[unique_key] for unique_key in unique_keys])
        # the appearances of nodes in nodes_neighbor_ids, ndarray, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, 2, self.walk_length + 1)
        node_appearances = unique_node_appearances[inverse_indices, :].reshape(nodes_neighbor_ids.shape[0], nodes_neighbor_ids.shape[1],
                                                                               nodes_neighbor_ids.shape[2], 2, self.walk_length + 1)

        # encode the node appearances in the random walks by MLPs
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, 2, position_feat_dim)
        position_features = self.position_encode_layer(torch.Tensor(node_appearances).float().to(self.device))
        # add the position features of each node in random walks generated by src and dst nodes by summing over the second last dimension, Equation (6) in CAWN paper
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, position_feat_dim)
        position_features = position_features.sum(dim=-2)
        return position_features

class BiLSTMEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        BiLSTM encoder.
        :param input_dim: int, dimension of the input
        :param hidden_dim: int, dimension of the hidden state
        """
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim_one_direction = hidden_dim // 2
        self.model_dim = self.hidden_dim_one_direction * 2
        self.bilstm_encoder = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_one_direction, batch_first=True, bidirectional=True)

    def forward(self, inputs: torch.Tensor, lengths: np.ndarray):
        """
        encode the inputs by BiLSTM encoder based on lengths
        :param inputs: Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.walk_length + 1, input_dim)
        :param lengths: ndarray, shape (batch_size, num_neighbors ** self.walk_length), record the valid length of each walk
        :return:
        """
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length), self.walk_length + 1, input_dim), which corresponds to the LSTM input (batch_size, seq_len, input_dim)
        inputs = inputs.reshape(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # a PackedSequence object, pack the padded sequence for efficient computation and avoid the errors of computing padded value, set enforce_sorted to False
        inputs = pack_padded_sequence(inputs, lengths.flatten(), batch_first=True, enforce_sorted=False)
        # the outputs of LSTM are output, (h_n, c_n), and we only use the output and do not use hidden states
        encoded_features, _ = self.bilstm_encoder(inputs)
        # encoded_features, Tensor, shape (batch_size * (num_neighbors ** self.walk_length), self.walk_length + 1, self.model_dim), pad the packed sequence
        # seq_lengths, Tensor, shape (batch_size * (num_neighbors ** self.walk_length), )
        encoded_features, seq_lengths = pad_packed_sequence(encoded_features, batch_first=True)
        assert (seq_lengths.numpy() == lengths.flatten()).all()
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length), ), the shifted sequence lengths
        shifted_seq_lengths = seq_lengths + torch.tensor([i * encoded_features.shape[1] for i in range(encoded_features.shape[0])])
        # Tensor, shape (batch_size * (num_neighbors ** self.walk_length) * (self.walk_length + 1), self.model_dim)
        encoded_features = encoded_features.reshape(encoded_features.shape[0] * encoded_features.shape[1], encoded_features.shape[2])
        # Tensor, shape (batch_size, num_neighbors ** self.walk_length, self.model_dim), get the encodings of each walk at the last position
        # note that we need to use shifted_seq_lengths - 1 to get the shifted indices
        encoded_features = encoded_features[shifted_seq_lengths - 1].reshape(lengths.shape[0], lengths.shape[1], self.model_dim)

        return encoded_features

# class leakchecker(nn.Module):
#     def __init__(self, device='cpu'):
#         super(leakchecker, self).__init__()
        