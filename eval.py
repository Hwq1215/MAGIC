import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import os
import umap

#预测值preds
def plot_scatter(x_train, x_test, y_test, n_node_feat,output_dir,preds=None):

    # # 创建训练数据的散点图
    # print("plot train drawing...")
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c='blue', label='Training Nodes', alpha=0.5)
    # plt.title('Training Data Scatter Plot')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.legend()

    # 创建测试数据的散点图
    # 保留 10% 的测试数据
    sample_indices = np.random.choice(len(x_test), size=int(0.1 * len(x_test)), replace=False)
    x_test_sampled = x_test[sample_indices]
    y_test_sampled = y_test[sample_indices]
    preds_sampled = preds[sample_indices]
    # 使用 UMAP 进行降维
    reducer = umap.UMAP(random_state=42)
    x_test_umap = reducer.fit_transform(x_test_sampled)

    # 创建测试数据的散点图
    print("plot test drawing...")
    plt.figure(figsize=(12, 6))

    if preds is not None and len(preds_sampled) == len(x_test_sampled):
        # 区分预测正确的样本和预测错误的样本
        correct_benign = (preds_sampled == 0.0) & (y_test_sampled == 0.0)
        correct_malicious = (preds_sampled == 1.0) & (y_test_sampled == 1.0)
        incorrect_benign = (preds_sampled == 1.0) & (y_test_sampled == 0.0)
        incorrect_malicious = (preds_sampled == 0.0) & (y_test_sampled == 1.0)

        # 绘制预测正确的 Benign 样本
        plt.scatter(x_test_umap[correct_benign, 0], x_test_umap[correct_benign, 1], c='green', marker='o', label='Correct Benign', alpha=0.5, s=2)

        # 绘制预测正确的 Malicious 样本
        plt.scatter(x_test_umap[correct_malicious, 0], x_test_umap[correct_malicious, 1], c='red', marker='o', label='Correct Malicious', alpha=0.5, s=2)

        # 绘制预测错误的 Benign 样本
        plt.scatter(x_test_umap[incorrect_benign, 0], x_test_umap[incorrect_benign, 1], c='green', marker='x', label='Incorrect Benign', alpha=0.5, s=2)

        # 绘制预测错误的 Malicious 样本
        plt.scatter(x_test_umap[incorrect_malicious, 0], x_test_umap[incorrect_malicious, 1], c='red', marker='x', label='Incorrect Malicious', alpha=0.5, s=2)
    else:
        # 分别为 Benign 和 Malicious 创建散点图
        benign_indices = [i for i in range(len(y_test_sampled)) if y_test_sampled[i] == 0.0]
        malicious_indices = [i for i in range(len(y_test_sampled)) if y_test_sampled[i] == 1.0]
        plt.scatter(x_test_umap[benign_indices, 0], x_test_umap[benign_indices, 1], c='green', label='Benign', alpha=0.5, s=1)
        plt.scatter(x_test_umap[malicious_indices, 0], x_test_umap[malicious_indices, 1], c='red', label='Malicious', alpha=0.5, s=1)
    plt.title('Test Data Scatter Plot')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(['Benign', 'Malicious'])

    plt.tight_layout()
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, 'hot_map.png'))
    plt.close()
def main(main_args):
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device=device)
    dataset_name = main_args.dataset
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        pooler = Pooling(main_args.pooling)
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, main_args.n_dim,
                                                    main_args.e_dim)
    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))
        model = model.to(device)
        model.eval()
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        with torch.no_grad():
            x_train = []
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            x_train = np.concatenate(x_train, axis=0)
            print(x_train.shape)
            skip_benign = 0
            x_test = []
            for i in range(n_test):
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)
                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()
                print(skip_benign)
                x_test.append(model.embed(g).cpu().numpy())
                del g
            x_test = np.concatenate(x_test, axis=0)
            print(x_test.shape)
            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0
            malicious_dict = {}
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Exclude training samples from the test set
            print(f"skip_benign: {skip_benign},x_test.shape[0]: {x_test.shape[0]}")
            test_idx = []
            for i in range(x_test.shape[0]):
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test
            test_auc, test_std, _, pred_test = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test,
                                                                       result_y_test)
            
            #绘制散点图
            output_dir = f"./plots/{dataset_name}"
            plot_scatter(x_train, result_x_test, result_y_test, main_args.n_dim,output_dir,preds=pred_test)
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
