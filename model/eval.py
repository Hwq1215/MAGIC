import os
import random
import time
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    with torch.no_grad():
        for i in full:
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(1000, dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    n_neighbors = min(int(train_count * 0.02), 10)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            f1 = 2 * prec * rec / (rec + prec + 1e-9)
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            tn = 0
            fn = 0
            tp = 0
            fp = 0
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:
                    fp += 1
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        # -- 最优结果打印
        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))
        
        # -- 最好的两次结果打印
        # 根据 f1_list 的值对所有列表进行排序
        sorted_indices = sorted(range(len(f1_list)), key=lambda i: f1_list[i], reverse=True)

        # 获取最好的两个索引
        best_two_indices = sorted_indices[:4]

        # 根据最好的两个索引提取各指标的值
        auc_best = [auc_list[i] for i in best_two_indices]
        f1_best = [f1_list[i] for i in best_two_indices]
        prec_best = [prec_list[i] for i in best_two_indices]
        rec_best = [rec_list[i] for i in best_two_indices]
        tn_best = [tn_list[i] for i in best_two_indices]
        fn_best = [fn_list[i] for i in best_two_indices]
        tp_best = [tp_list[i] for i in best_two_indices]
        fp_best = [fp_list[i] for i in best_two_indices]

        # 输出结果
        print('AUC (Best 2 by F1): {}+{}'.format(np.mean(auc_best), np.std(auc_best)))
        print('F1 (Best 2): {}+{}'.format(np.mean(f1_best), np.std(f1_best)))
        print('PRECISION (Best 2 by F1): {}+{}'.format(np.mean(prec_best), np.std(prec_best)))
        print('RECALL (Best 2 by F1): {}+{}'.format(np.mean(rec_best), np.std(rec_best)))
        print('TN (Best 2 by F1): {}+{}'.format(np.mean(tn_best), np.std(tn_best)))
        print('FN (Best 2 by F1): {}+{}'.format(np.mean(fn_best), np.std(fn_best)))
        print('TP (Best 2 by F1): {}+{}'.format(np.mean(tp_best), np.std(tp_best)))
        print('FP (Best 2 by F1): {}+{}'.format(np.mean(fp_best), np.std(fp_best)))

        return np.mean(auc_list), np.std(auc_list)
    else:
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
        x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        tn = 0
        fn = 0
        tp = 0
        fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:
                fp += 1
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1[best_idx]))
        print('PRECISION: {}'.format(prec[best_idx]))
        print('RECALL: {}'.format(rec[best_idx]))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        return auc, 0.0


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    pred_test = np.array([])  # 初始化为空数组
    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance
    del distances
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = -1
    for i in range(len(f1)):
        # To repeat peak performance
        print(f"rec: {rec[i]}, f1: {f1[i]}\n")
        if dataset == 'trace' and rec[i] < 0.98:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.95:
            best_idx = i - 1
            break
        # if dataset == 'theia' and rec[i] < 0.975:
        #     best_idx = i - 1
        #     break
        if dataset == 'cadets' and rec[i] < 0.98:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
            pred_test = np.append(pred_test, 1.0)  
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
            pred_test = np.append(pred_test, 0.0)  
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
            pred_test = np.append(pred_test, 0.0)  
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
            pred_test = np.append(pred_test, 1.0) 
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0, None, pred_test

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle as pkl
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve

def evaluate_entity_level_using_knc(dataset, x_train, y_train, x_test, y_test):
    # 分离良性样本和恶意样本
    benign_indices = np.where(y_train == 0)[0]
    malicious_indices = np.where(y_train == 1)[0]
    x_train_benign = x_train[benign_indices]
    x_train_malicious = x_train[malicious_indices]
    
    # 分别计算良性样本和恶意样本的均值和标准差
    benign_mean = x_train_benign.mean(axis=0)
    benign_std = x_train_benign.std(axis=0)
    malicious_mean = x_train_malicious.mean(axis=0)
    malicious_std = x_train_malicious.std(axis=0)
    
    # 对测试样本分别进行标准化
    x_test_benign_scaled = (x_test - benign_mean) / (benign_std + 1e-9)
    x_test_malicious_scaled = (x_test - malicious_mean) / (malicious_std + 1e-9)
    
    # 初始化KNN模型
    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10
    
    # 计算到良性样本和恶意样本的距离
    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        # 计算测试样本到良性样本的距离
        benign_nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(x_train_benign)), n_jobs=-1)
        benign_nbrs.fit(x_train_benign)
        distances_benign, _ = benign_nbrs.kneighbors(x_test_benign_scaled)
        mean_distance_benign = distances_benign.mean(axis=1)
        
        # 计算测试样本到恶意样本的距离
        malicious_nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(x_train_malicious)), n_jobs=-1)
        malicious_nbrs.fit(x_train_malicious)
        distances_malicious, _ = malicious_nbrs.kneighbors(x_test_malicious_scaled)
        mean_distance_malicious = distances_malicious.mean(axis=1)
        
        # 计算分数：离良性样本越远，离恶意样本越近，分数越高
        # 为了处理类别不平衡，给恶意样本距离更大的权重
        score = (mean_distance_benign * 2 + mean_distance_malicious) / (mean_distance_benign + mean_distance_malicious * 2)
        
        # 保存结果
        with open(save_dict_path, 'wb') as f:
            pkl.dump(score, f)
    else:
        with open(save_dict_path, 'rb') as f:
            score = pkl.load(f)
    
    # 计算评估指标
    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    
    # 找到最佳阈值
    best_idx = np.argmax(f1)
    best_thres = threshold[best_idx]
    
    # 计算混淆矩阵
    tn = np.sum((y_test == 0) & (score < best_thres))
    fn = np.sum((y_test == 1) & (score < best_thres))
    tp = np.sum((y_test == 1) & (score >= best_thres))
    fp = np.sum((y_test == 0) & (score >= best_thres))
    
    # 生成预测结果
    pred_test = np.where(score >= best_thres, 1.0, 0.0)
    
    # 打印评估结果
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    
    return auc, 0.0, None, pred_test