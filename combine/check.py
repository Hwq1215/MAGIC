import argparse
import json
import pickle as pkl
import networkx as nx

def verify_train_graphs(dataset):
    # 加载训练图
    train_gs = [nx.node_link_graph(data) for data in pkl.load(open(f'../data/{dataset}/train.pkl', 'rb'))]
    
    # 加载节点类型 ID 映射
    with open(f'../data/{dataset}/node_type_id.txt', 'r', encoding='utf-8') as f:
        node_type_id_map = json.load(f)
    
    # 创建节点类型映射的逆映射
    node_id_type_map = {v: k for k, v in node_type_id_map.items()}
    
    # 加载恶意节点
    with open(f'../data/{dataset}/malicious.pkl', 'rb') as f:
        malicious_entities, _ = pkl.load(f)
    
    # 验证每个图中的节点类型
    for i, train_g in enumerate(train_gs):
        for node in train_g.nodes():
            node_type_id = train_g.nodes[node]['type']
            node_type = node_id_type_map.get(node_type_id, 'Unknown')
            if node in malicious_entities and node_type != 'MemoryObject':
                print(f"警告: 训练集图 {i} 中包含非 MemoryObject 类型的恶意节点 {node}，类型为 {node_type}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='验证训练集中的节点类型')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError
    
    verify_train_graphs(args.dataset)