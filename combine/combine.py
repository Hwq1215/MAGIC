import argparse
import networkx as nx
import pickle as pkl
import json

def get_malicious_graphs(dataset, train_gs, test_gs, malicious_entities):
    malicious_train_gs = []
    malicious_test_gs = []
    malicious_node_ids_train = []  # 存储恶意节点的ID
    malicious_node_ids_test = []  # 存储恶意节点的ID

    # 提取训练图中的恶意子图
    for train_g in train_gs:
        malicious_nodes = []
        for node in train_g.nodes():
            # 尝试从节点属性中获取类型，如果不存在默认为 -1
            node_type = train_g.nodes[node].get('type', -1)
            if node in malicious_entities and node_type != 2:
                malicious_nodes.append(node)
        malicious_train_g = train_g.subgraph(malicious_nodes)
        malicious_train_gs.append(malicious_train_g)
        malicious_node_ids_train.extend(malicious_nodes)  # 记录恶意节点的ID

    # 提取测试图中的恶意子图
    for test_g in test_gs:
        malicious_nodes = []
        for node in test_g.nodes():
            # 尝试从节点属性中获取类型，如果不存在默认为 -1
            node_type = test_g.nodes[node].get('type', -1)
            if node in malicious_entities and node_type != 2:
                malicious_nodes.append(node)
        malicious_test_g = test_g.subgraph(malicious_nodes)
        malicious_test_gs.append(malicious_test_g)
        malicious_node_ids_test.extend(malicious_nodes)  # 记录恶意节点的ID

    return malicious_train_gs, malicious_test_gs, malicious_node_ids_train, malicious_node_ids_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError
    
    # 加载训练图和测试图
    train_gs = [nx.node_link_graph(data) for data in pkl.load(open(f'../data/{args.dataset}/train.pkl', 'rb'))]
    test_gs = [nx.node_link_graph(data) for data in pkl.load(open(f'../data/{args.dataset}/test.pkl', 'rb'))]
    malicious_entities = set(pkl.load(open(f'../data/{args.dataset}/malicious.pkl', 'rb'))[0])

    # 获取恶意图和恶意节点ID
    malicious_train_gs, malicious_test_gs, malicious_node_ids_train, malicious_node_ids_test = get_malicious_graphs(args.dataset, train_gs, test_gs, malicious_entities)

    # 保存恶意图
    pkl.dump([nx.node_link_data(g) for g in malicious_train_gs], open(f'../data/{args.dataset}/malicious_train.pkl', 'wb'))
    pkl.dump([nx.node_link_data(g) for g in malicious_test_gs], open(f'../data/{args.dataset}/malicious_test.pkl', 'wb'))

    # 保存恶意节点的ID到文件
    with open(f'../data/{args.dataset}/malicious_node_ids_train.txt', 'w') as f:
        for node_id in malicious_node_ids_train:
            f.write(f"{node_id}\n")
    
    with open(f'../data/{args.dataset}/malicious_node_ids_test.txt', 'w') as f:
        for node_id in malicious_node_ids_test:
            f.write(f"{node_id}\n")

    # 打印保存路径
    print(f"恶意节点的ID（训练集）已保存到: ../data/{args.dataset}/malicious_node_ids_train.txt")
    print(f"恶意节点的ID（测试集）已保存到: ../data/{args.dataset}/malicious_node_ids_test.txt")