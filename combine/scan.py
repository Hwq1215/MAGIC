import pickle as pkl
import argparse



def scan(args):
    # 加载 train.pkl 文件
    path = '../data/{}/train.pkl'.format(args.dataset)  # 替换为你的 dataset 名称，例如 'trace'
    with open(path, 'rb') as f:
        train_data = pkl.load(f)

    # 查看 train_data 的内容
    print(train_data)  # 打印整个内容
    print(type(train_data))  # 查看数据类型
    print(len(train_data))  # 查看有多少个图
    print(train_data[0])  # 查看第一个图的数据结构

    # 加载 train.pkl 文件
    path = '../data/{}/train.pkl'.format(args.dataset)  # 替换为你的 dataset 名称，例如 'trace'
    with open(path, 'rb') as f:
        train_data = pkl.load(f)

    # 查看 train_data 的内容
    print(train_data)  # 打印整个内容
    print(type(train_data))  # 查看数据类型
    print(len(train_data))  # 查看有多少个图
    print(train_data[0])  # 查看第一个图的数据结构

    import networkx as nx

    # 将序列化数据转换为图对象
    graph = nx.node_link_graph(train_data[0])

    # 查看图的节点和边
    print("Nodes:", graph.nodes(data=True))
    print("Edges:", graph.edges(data=True))

    # for graph_data in train_data[0]:
    #     print(f"Graph {i+1}:")
    #     print("Nodes:", graph_data['nodes'])
    #     print("Edges:", graph_data['links'])
    #     print("Node Attributes:", graph_data.get('node_attrs', None))
    #     print("Edge Attributes:", graph_data.get('edge_attrs', None))
    #     print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    scan(parser.parse_args())