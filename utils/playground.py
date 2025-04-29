import argparse
import json
import os
import random
import re

from tqdm import tqdm
import networkx as nx
import pickle as pkl


node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0

metadata = {
    'trace':{
        'test': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4']
    },
    'theia':{
            'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets':{
            'test': ['ta1-cadets-e3-official-2.json']
    }
}


pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')

def extract_malicious_connections(test_gs, final_malicious_entities, dataset):
    """
    提取恶意图中的连接关系并保存到文件。
    
    :param test_gs: 测试图列表
    :param final_malicious_entities: 恶意节点列表
    :param dataset: 数据集名称
    """
    malicious_connections = []

    for test_g in test_gs:
        # 提取测试图中的恶意节点
        malicious_nodes = [node for node in test_g.nodes() if node in final_malicious_entities]
        if not malicious_nodes:
            continue
        
        # 创建包含恶意节点的子图
        malicious_subgraph = test_g.subgraph(malicious_nodes).copy()
        
        # 提取恶意节点之间的边
        for src, dst, data in malicious_subgraph.edges(data=True):
            malicious_connections.append({
                "src": src,
                "dst": dst,
                "edge_type": list(edge_type_dict.keys())[list(edge_type_dict.values()).index(data['type'])]
            })
    
    # 保存恶意连接关系到JSON文件
    if malicious_connections:
        with open(f'../data/{dataset}/malicious_connections.txt', 'w', encoding='utf-8') as f:
            json.dump(malicious_connections, f, indent=4)
        print(f"已将恶意图的连接关系保存到 ../data/{dataset}/malicious_connections.txt")

def extract_attack_paths(dataset, test_gs, final_malicious_entities):
    """
    提取攻击路径并划分为图。
    
    :param dataset: 数据集名称
    :param test_gs: 测试图列表
    :param final_malicious_entities: 恶意节点列表
    """
    attack_path_graphs = []
    malicious_node_info = []

    for idx, test_g in enumerate(test_gs):
        # 提取测试图中的恶意节点
        malicious_nodes = [node for node in test_g.nodes() if node in final_malicious_entities]
        if not malicious_nodes:
            continue
            
        # 创建包含恶意节点的子图
        malicious_subgraph = test_g.subgraph(malicious_nodes).copy()
        
        # 将子图划分为多个连通分量（适用于有向图）
        connected_components = list(nx.weakly_connected_components(malicious_subgraph))
        
        # 为每个连通分量创建独立的图并保存到数组中
        for i, component in enumerate(connected_components):
            subgraph = malicious_subgraph.subgraph(component).copy()
            subgraph.graph['dataset'] = 'attack_path'
            subgraph.graph['id'] = f"{idx}_{i}"
            attack_path_graphs.append(subgraph)
            
            # 保存节点信息
            node_info = {
                "counts": len(component),
                "nodes": list(component)
            }
            malicious_node_info.append(node_info)
    
    # 保存所有离散的攻击路径图到一个数组中
    if attack_path_graphs:
        pkl.dump(attack_path_graphs, open(f'../data/{dataset}/attack_path_graphs.pkl', 'wb'))
        print(f"已将所有离散的攻击路径图保存到 ../data/{dataset}/attack_path_graphs.pkl")
    
    # 保存恶意节点信息到JSON文件
    if malicious_node_info:
        with open(f'../data/{dataset}/malicious_nodes_info.txt', 'w', encoding='utf-8') as f:
            json.dump(malicious_node_info, f, indent=4)
        print(f"已将恶意节点信息保存到 ../data/{dataset}/malicious_nodes_info.txt")

def create_random_malicious_graph(test_gs, final_malicious_entities, ratio, dataset):
    selected_malicious_entities = []

    for test_g in test_gs:
        visited = {node: False for node in test_g.nodes()}
        malicious_nodes = [node for node in test_g.nodes() if node in final_malicious_entities]
        if not malicious_nodes:
            continue

        current_nodes = []
        remaining_ratio = ratio

        while len(current_nodes) < int(len(final_malicious_entities) * ratio):
            available_start_nodes = [node for node in malicious_nodes if not visited[node]]
            if not available_start_nodes:
                break
            start_node = random.choice(available_start_nodes)
            stack = [start_node]
            visited[start_node] = True

            while stack:
                current_node = stack.pop()
                current_nodes.append(current_node)
                if len(current_nodes) >= int(len(final_malicious_entities) * ratio):
                    break
                # 获取正向和反向的邻居
                for neighbor in get_all_neighbors(test_g, current_node):
                    if neighbor in final_malicious_entities and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            if current_nodes:
                break

        selected_malicious_entities.extend(current_nodes)

    new_graphs = []
    for test_g in test_gs:
        malicious_nodes = [node for node in test_g.nodes() if node in selected_malicious_entities]
        if malicious_nodes:
            malicious_subgraph = test_g.subgraph(malicious_nodes).copy()
            new_graphs.append(malicious_subgraph)
    
    if new_graphs:
        pkl.dump([nx.node_link_data(new_g) for new_g in new_graphs], open(f'../data/{dataset}/random_malicious_graphs.pkl', 'wb'))
        print(f"已将随机恶意图保存到 ../data/{dataset}/random_malicious_graphs.pkl")
    
    with open(f'../data/{dataset}/random_malicious.txt', 'w', encoding='utf-8') as f:
        for node_id in selected_malicious_entities:
            f.write(f'{node_id}\n')
    print(f"已将随机恶意节点信息保存到 ../data/{dataset}/random_malicious.txt")


def get_all_neighbors(graph, node):
    """
    获取节点的正向和反向邻居。

    :param graph: 图
    :param node: 节点
    :return: 正向和反向邻居的列表
    """
    # 正向邻居
    successors = list(graph.neighbors(node))
    # 反向邻居（前驱节点）
    predecessors = list(graph.predecessors(node))
    # 合并邻居列表
    all_neighbors = successors + predecessors
    return all_neighbors

def read_single_graph(dataset, malicious, path, test=False):
    global node_type_cnt, edge_type_cnt
    g = nx.DiGraph()
    print('converting {} ...'.format(path))
    path = '../data/{}/'.format(dataset) + path + '.txt'
    with open(path, 'r') as f:
        lines = []
        for line in f:
            src, src_type, dst, dst_type, edge_type, ts = line.strip().split('\t')
            ts = int(ts)
            if not test:
                if src in malicious or dst in malicious:
                    if src in malicious and src_type != 'MemoryObject':
                        continue
                    if dst in malicious and dst_type != 'MemoryObject':
                        continue
            
            if src_type not in node_type_dict:
                node_type_dict[src_type] = node_type_cnt
                node_type_cnt += 1
            if dst_type not in node_type_dict:
                node_type_dict[dst_type] = node_type_cnt
                node_type_cnt += 1
            if edge_type not in edge_type_dict:
                edge_type_dict[edge_type] = edge_type_cnt
                edge_type_cnt += 1
            
            if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
                lines.append([dst, src, dst_type, src_type, edge_type, ts])
            else:
                lines.append([src, dst, src_type, dst_type, edge_type, ts])
    
    lines.sort(key=lambda x: x[5])
    
    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []
    for line in lines:
        src, dst, src_type, dst_type, edge_type = line[:5]
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]
        
        if src not in node_map:
            node_map[src] = node_cnt
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        if dst not in node_map:
            node_map[dst] = node_cnt
            g.add_node(node_cnt, type=dst_type_id)
            node_list.append(dst)
            node_type_map[dst] = dst_type
            node_cnt += 1
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)
    
    return node_map, g


def preprocess_dataset(dataset):
    id_nodetype_map = {}
    id_nodename_map = {}
    for file in os.listdir(f'../data/{dataset}'):
        if 'json' in file and not '.txt' in file and not 'names' in file and not 'types' in file and not 'metadata' in file and not 'tar.gz' in file:
            print(f'reading {file} ...')
            with open(f'../data/{dataset}/{file}', 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: 
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: 
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: 
                        continue
                    if len(pattern_uuid.findall(line)) == 0: 
                        print(line)
                    uuid = pattern_uuid.findall(line)[0]
                    subject_type = pattern_type.findall(line)

                    if len(subject_type) < 1:
                        if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                            subject_type = 'MemoryObject'
                        if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                            subject_type = 'NetFlowObject'
                        if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                            subject_type = 'UnnamedPipeObject'
                    else:
                        subject_type = subject_type[0]

                    if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                        continue
                    id_nodetype_map[uuid] = subject_type
                    if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                        id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                    elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                        id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                    elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                        id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            if os.path.exists(f'../data/{dataset}/{file}.txt'):
                continue
            with open(f'../data/{dataset}/{file}', 'r', encoding='utf-8') as f:
                with open(f'../data/{dataset}/{file}.txt', 'w', encoding='utf-8') as fw:
                    print(f'processing {file} ...')
                    for line in tqdm(f):
                        if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                            edgeType = pattern_type.findall(line)[0]
                            timestamp = pattern_time.findall(line)[0]
                            srcId = pattern_src.findall(line)

                            if len(srcId) == 0: 
                                continue
                            srcId = srcId[0]
                            if srcId not in id_nodetype_map:
                                continue
                            srcType = id_nodetype_map[srcId]
                            dstId1 = pattern_dst1.findall(line)
                            if len(dstId1) > 0 and dstId1[0] != 'null':
                                dstId1 = dstId1[0]
                                if dstId1 not in id_nodetype_map:
                                    continue
                                dstType1 = id_nodetype_map[dstId1]
                                this_edge1 = f'{srcId}\t{srcType}\t{dstId1}\t{dstType1}\t{edgeType}\t{timestamp}\n'
                                fw.write(this_edge1)

                            dstId2 = pattern_dst2.findall(line)
                            if len(dstId2) > 0 and dstId2[0] != 'null':
                                dstId2 = dstId2[0]
                                if dstId2 not in id_nodetype_map:
                                    continue
                                dstType2 = id_nodetype_map[dstId2]
                                this_edge2 = f'{srcId}\t{srcType}\t{dstId2}\t{dstType2}\t{edgeType}\t{timestamp}\n'
                                fw.write(this_edge2)
    if id_nodename_map:
        with open(f'../data/{dataset}/names.json', 'w', encoding='utf-8') as fw:
            json.dump(id_nodename_map, fw)
    if id_nodetype_map:
        with open(f'../data/{dataset}/types.json', 'w', encoding='utf-8') as fw:
            json.dump(id_nodetype_map, fw)


def read_graphs(dataset):
    malicious_entities = set()
    malicious_path = f'../data/{dataset}/{dataset}.txt'
    if os.path.exists(malicious_path):
        with open(malicious_path, 'r') as f:
            for line in f:
                malicious_entities.add(line.strip())

    preprocess_dataset(dataset)
    
    test_gs = []
    test_node_map = {}
    count_node = 0
    for file in metadata[dataset]['test']:
        node_map, test_g = read_single_graph(dataset, malicious_entities, file, True)
        if test_g.number_of_nodes() > 0:
            test_g.graph['dataset'] = 'test'
            test_gs.append(test_g)
            for key in node_map:
                if key not in test_node_map:
                    test_node_map[key] = node_map[key] + count_node
            count_node += test_g.number_of_nodes()

    if os.path.exists(f'../data/{dataset}/names.json') and os.path.exists(f'../data/{dataset}/types.json'):
        with open(f'../data/{dataset}/names.json', 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)
        with open(f'../data/{dataset}/types.json', 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)
        with open(f'../data/{dataset}/malicious_names.txt', 'w', encoding='utf-8') as f:
            final_malicious_entities = []
            malicious_names = []
            for e in malicious_entities:
                if e in test_node_map and e in id_nodetype_map and id_nodetype_map[e] != 'MemoryObject' and id_nodetype_map[e] != 'UnnamedPipeObject':
                    final_malicious_entities.append(test_node_map[e])
                    if e in id_nodename_map:
                        malicious_names.append(id_nodename_map[e])
                        f.write(f'{e}\t{id_nodename_map[e]}\n')
                    else:
                        malicious_names.append(e)
                        f.write(f'{e}\t{e}\n')
    else:
        with open(f'../data/{dataset}/malicious_names.txt', 'w', encoding='utf-8') as f:
            final_malicious_entities = []
            malicious_names = []
            for e in malicious_entities:
                if e in test_node_map:
                    final_malicious_entities.append(test_node_map[e])
                    malicious_names.append(e)
                    f.write(f'{e}\t{e}\n')

    # 保存恶意节点的 UUID 和名称
    with open(f'../data/{dataset}/malicious.pkl', 'wb') as f:
        pkl.dump((final_malicious_entities, malicious_names), f)

    # 创建随机恶意图
    create_random_malicious_graph(test_gs, final_malicious_entities, 0.8, dataset)

    # 保存测试集
    with open(f'../data/{dataset}/test.pkl', 'wb') as f:
        pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], f)

    # 保存节点类型 ID 映射
    with open(f'../data/{dataset}/node_type_id.txt', 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in node_type_dict.items()}, f, indent=4)

    # 保存边类型 ID 映射
    with open(f'../data/{dataset}/edge_type_id.txt', 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in edge_type_dict.items()}, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()
    
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError
    
    read_graphs(args.dataset)