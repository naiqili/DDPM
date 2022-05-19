import os, time, datetime
import sys
import torch
import torch.optim as optim
from tqdm import tqdm, trange
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import argparse
from argparse import Namespace

from nice import NICE
from dircluster import sample_mu_lam, SSE, LLH
from utils import savefig_clusters, mvnlogpdf, plt2img
from AE import AE

import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
from numpy import log, exp, pi
from scipy.stats import wishart, gamma
from scipy.stats import multivariate_normal as normal
from numpy.linalg import inv, det
from matplotlib.patches import Ellipse
from scipy.special import loggamma
import tensorflow as tf
from scipy.stats import wishart, gamma, entropy
# from plot_cf import pretty_plot_confusion_matrix
import pandas as pd
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, pair_confusion_matrix, normalized_mutual_info_score, adjusted_mutual_info_score
# matplotlib.use('Agg')
from ae_util import *
from utils import *


TRAIN_BATCH_SIZE = 128


def parse_args(manual_args=None):
    parser = argparse.ArgumentParser(description='main.')
    # dataset params
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dim', type=int, default=10)  # todo 以前默认是768， 会默认算到超参里面去，影响？
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--n_sample_load', type=int, default=10000)

    # n_iter params
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--iter_dmm', type=float, default=[2, 1, 0.5], nargs='*')
    parser.add_argument('--dmm_rebuild_freq', type=int, default=50)
    parser.add_argument('--iter_nice', type=float, default=[2, 1, 0.5], nargs='*')
    parser.add_argument('--save_freq', type=int, default=20)

    # core hyper parameters
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--maxK', type=int, default=500)
    parser.add_argument('--logalpha', type=float, default=0.0)
    parser.add_argument('--a0', type=float, default=50.0)
    parser.add_argument('--b0', type=float, default=10.0)
    parser.add_argument('--kappa0', type=float, default=1.0)

    parser.add_argument('--decaycoef', type=float, default=0.9)
    parser.add_argument('--decaymin', type=float, default=0.1)
    parser.add_argument('--contralam', type=float, default=1.0)
    parser.add_argument('--reglam', type=float, default=1.0)
    parser.add_argument('--hinge', type=float, default=100.0)

    parser.add_argument('--figpath', type=str, default='./res2')
    parser.add_argument('--pretrainpath', type=str, default='./saved_ae/mnist_ae_best428.pt')
    parser.add_argument('--aex_file', type=str, default='./saved_ae/mnist_ae_feat.pt')

    # additional params
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--a_stage_steps', type=int, default=[-1, -1], nargs='*')
    parser.add_argument('--a_stage_values', type=float, default=[0, 0], nargs='*')
    parser.add_argument('--with_flow', type=str, default='yes')
    parser.add_argument('--restart_dmm', type=str, default='no')
    parser.add_argument('--restart_flow', type=str, default='no')
    parser.add_argument('--refresh_cc_in_flow', type=str, default='no')
    parser.add_argument('--grad_clip', type=float, default=1)  # grad norm clipping
    parser.add_argument('--loss_clip', type=float, default=1)  # grad norm clipping
    parser.add_argument('--log_dir', type=str, default='./exp_out')

    parser.add_argument('--nice_nlayers', type=int, default=6)
    parser.add_argument('--nice_units', type=int, default=512)
    parser.add_argument('--input_normalize', type=str, default='no')
    parser.add_argument('--device', type=str, default='cpu')
    args, args_list = parser.parse_known_args(args=manual_args)

    # param check
    args.input_normalize = True if args.input_normalize == 'yes' else False
    assert len(args.a_stage_steps) == len(args.a_stage_values)
    args.a_stage_steps = np.asarray(args.a_stage_steps)
    args.a_stage_values = np.asarray(args.a_stage_values)

    # init necessary arguments
    if not os.path.exists(args.figpath):
        os.makedirs(args.figpath)

    print(args)
    return args


def get_iter_num(n_epoch, n_samples, cfg_list):
    '''
    How many steps for iteration in the n-th epoch
    num_iter is configed to decay for fast training 
    '''
    ratio = cfg_list[-1] if n_epoch >= len(cfg_list) else cfg_list[n_epoch]
    return int(np.ceil(n_samples * ratio))


def cluster2label(K, samples_k, y):
    map_dict = {}
    y_pred = samples_k.copy()
    for k_idx in range(K):  # analysis each cluster
        cluster_sid = (samples_k == k_idx)
        label, counts = np.unique(y[cluster_sid], return_counts=True)
        major_class_idx = np.argmax(counts)

        map_dict[k_idx] = label[major_class_idx]
        y_pred[cluster_sid] = label[major_class_idx]
    return y_pred, map_dict


def pair_f1_score(ys, ks):
    cf = pair_confusion_matrix(ys, ks)
    # TN, FP
    # FN, TP
    return cf[1, 1] / float(cf[1, 1] + 0.5 * (cf[0, 1] + cf[1, 0]))


def get_count_matrix(K, cluster_label, nC, label):
    # count labels in each cluster
    np.unique(cluster_label)
    cluster_counts_list = []
    for k in range(K):
        c_sample_id = (cluster_label == k)
        y, ny = np.unique(label[c_sample_id], return_counts=True)
        counts_arr = np.zeros(nC)
        counts_arr[y] = ny
        cluster_counts_list.append(counts_arr)
    count_matrix = np.stack(cluster_counts_list, axis=0)
    return count_matrix


def cluster_acc_fixed(Y_pred, Y):  # from paper VaDE code
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    a, b = linear_assignment(w.max() - w)
    ind = [(a[i], b[i]) for i in range(a.shape[0])]
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, ind


def eval_cluster(samples_k, ys, return_extra=False, alg='default', ds_name='default'):
    cids, nK = np.unique(samples_k, return_counts=True)
    k_indices = np.argsort(-nK)  # desc sort by size
    k_major, k_entropy = [], []
    confusion_matrix = np.zeros(shape=(10, 10))
    correct = 0
    for k_idx in k_indices:  # analysis each cluster
        cluster_sid = (samples_k == cids[k_idx])
        cluster_size = nK[k_idx]
        label, counts = np.unique(ys[cluster_sid], return_counts=True)
        k_entropy.append(entropy(counts / cluster_size))

        major_class_idx = np.argmax(counts)
        k_major.append(label[major_class_idx])
        confusion_matrix[label, label[major_class_idx]] += counts
        correct += counts[major_class_idx]
    acc = correct / np.sum(nK)
    weighted_entropy = np.sum(nK[k_indices]/np.sum(nK) * k_entropy)

    # k-irrevlent measure
    v_score, adj_rand = v_measure_score(ys, samples_k), adjusted_rand_score(ys, samples_k),
    pair_f1 = pair_f1_score(ys, samples_k)
    if not return_extra:
        # print(f'{alg} on {ds_name}:\n\t F score | V score | ARI | ACC* | K | wEntropy :\n\t'
        #       f'{pair_f1:.4f} & {v_score:.4f} & {adj_rand:.4f} & {acc:.4f} & {len(cids)} & {weighted_entropy:.4f}')
        return acc, v_score, adj_rand, pair_f1, weighted_entropy, confusion_matrix
    else:
        nmi = normalized_mutual_info_score(ys, samples_k)
        # print(f'{alg} on {ds_name}:\n\t F score | V score | ARI | NMI | ACC* | K | wEntropy :\n\t'
        #       f'{pair_f1:.4f} & {v_score:.4f} & {adj_rand:.4f} & {nmi:.4f} & {acc:.4f} & {len(cids)} & {weighted_entropy:.4f}')
        return pair_f1, v_score, adj_rand, nmi, acc, len(cids), weighted_entropy, confusion_matrix


def dirichlet_clustering(global_epoch, tb_writter, dir_params, samples, ys, n_iter_samples, args, log_step=0):
    # samples contains the full dataset, but used sample is indicated by n_sample_load
    # extract variables
    hp = dir_params.hyper
    _mu0, _ka0, logalpha, _a0, _b0 = hp.mu0, hp.ka0, hp.logalpha, hp.a0, hp.b0
    K, lam_K, mu_K, nK, ks = dir_params.K, dir_params.lam_K, dir_params.mu_K, dir_params.n_K, dir_params.samples_k

    # begin iter
    # random_samples_idx_subset = np.random.randint(0, len(ks), n_iter_samples)
    monitor_freq = 100
    with tb_writter.as_default():
        for iter_idx in range(n_iter_samples):
            sample_idx = iter_idx % len(ks)  # mod by n_sample_load
            # dynamic alpha
            if args.a_stage_steps[0] != -1:  # when enabled
                logalpha = args.a_stage_values[np.where(iter_idx >= args.a_stage_steps)[0][-1]]

            ## report result
            global_log_step = iter_idx+log_step

            # others
            if (global_log_step + 1) % monitor_freq == 0:
                # report cluster information
                tf.summary.scalar('dmm/K', K, step=iter_idx+log_step)
                arr = np.asarray(nK)/sum(nK)
                tf.summary.histogram('cluster size ratio', arr, step=global_log_step)
                mu_norm = np.linalg.norm(np.asarray(mu_K), axis=1)
                tf.summary.histogram('cluster mu norm', mu_norm, step=global_log_step)
                tf.summary.histogram('cluster lambda', np.asarray(lam_K), step=global_log_step)

                # report metrics
                if (global_log_step + 1) % (monitor_freq*10) == 0:
                    error = SSE(samples, K, ks, nK, mu_K, lam_K)
                    pair_f1, v_score, adj_rand, nmi, acc, _, w_ent, _ = eval_cluster(ks, ys,return_extra=True, ds_name=args.dataset)
                    tf.summary.scalar('dmm/sse', error, step=global_log_step)
                    tf.summary.scalar('dmm/acc', acc, step=global_log_step)
                    tf.summary.scalar('dmm/v_score', v_score, step=global_log_step)
                    tf.summary.scalar('dmm/nmi', nmi, step=global_log_step)
                    tf.summary.scalar('dmm/adj_rand', adj_rand, step=global_log_step)
                    tf.summary.scalar('dmm/pair_f1', pair_f1, step=global_log_step)
                    tf.summary.scalar('dmm/w_entropy', w_ent, step=global_log_step)
                    tb_writter.flush()
            # if (global_log_step + 1) % args.n_sample_load == 0:
            #     # show confusion matrix
            #     results = eval_cluster(nK, ks, ys)
            #     cf = results[-1]
            #     pllt, fig = pretty_plot_confusion_matrix(pd.DataFrame(cf, index=range(10), columns=range(10)))
            #     cf_img = plt2img(pllt, fig)
            #     tf.summary.image('confusion matrix', cf_img, step=global_log_step)
            #     # show top 15 clusters
            #     # img = savefig_clusters(global_epoch, K, ks, ds_ae, samples_flow_z,
            #     #                        mu_K, path=args.figpath, datasetname=args.dataset,
            #     #                        cluster_idx_lst=np.argsort(arr)[-1:-15:-1])
            #     # tf.summary.image('cluster info', img, step=global_log_step)
            #     tb_writter.flush()

            # executing DPM
            xi = samples[sample_idx]
            old_k = ks[sample_idx]
            if old_k != -1:
                # remove x[n], re-label if cluster becomes empty
                # 用最后一个位置填空位
                nK[old_k] -= 1
                if nK[old_k] == 0:  # 删除空簇
                    idx = (ks == K - 1)
                    ks[idx] = old_k
                    nK[old_k], lam_K[old_k], mu_K[old_k], = nK[K - 1], lam_K[K - 1], mu_K[K - 1]
                    nK, lam_K, mu_K = nK[:-1], lam_K[:-1], mu_K[:-1]
                    K -= 1

            # 3.1.1 Sampling Cluster Assingment k
            p_lst = []  # p_lst[k] is the probability that xi in C_k, p_lst[-1] means xi forms a new cluster
            Kn = min(K, args.maxK)  # 最大运算类别数限制，dirichlet可能刚开始生产的过多
            if K == 0:  # if no cluster, just build a new one
                chosen_k = 0
            else:
                # existing cluster assignment prob
                Klst = np.random.choice(K, size=Kn, replace=False, p=np.asarray(nK) / np.sum(np.asarray(nK)))
                for k in Klst:
                    pk = log(nK[k]) + mvnlogpdf(xi, mu_K[k], 1 / lam_K[k])
                    p_lst.append(pk)

                # new cluster assignment pob.
                _kan = _ka0 + 1
                _an = _a0 + 0.5 * args.dim
                _bn = _b0 + _ka0 * np.linalg.norm(xi - _mu0) ** 2 / (2 * (_ka0 + 1))
                logpk = log(logalpha) + \
                        loggamma(_an) - loggamma(_a0) + \
                        _a0 * log(_b0) - _an * log(_bn) + \
                        0.5 * (log(_ka0) - log(_kan)) - args.dim / 2 * log(2 * pi)
                p_lst.append(logpk)

                # sampling according to the assignment prob.
                maxpk = max(p_lst)
                p_lst = [exp(v - maxpk) for v in p_lst]  # 防止数值溢出
                chosen_k = np.random.choice(list(range(Kn + 1)),
                                            p=p_lst / sum(p_lst))  # sample, now xi belongs cluster chosen_k

            # 3.1.2 Sampling Cluster mean and lambda
            if chosen_k == Kn:  # assigned to new cluster
                nK.append(1)
                ks[sample_idx] = K
                lam_k, mu_k = sample_mu_lam(samples, nK, ks, chosen_k, _mu0, _ka0, _a0, _b0)
                mu_K.append(mu_k)
                lam_K.append(lam_k)
                K += 1
            else:  # assigned to existing cluster
                chosen_k = Klst[chosen_k]
                nK[chosen_k] += 1
                ks[sample_idx] = chosen_k
            # 过一段时间再实际更新簇中心点的参数
            if sample_idx % args.dmm_rebuild_freq == 0 or sample_idx == args.n_sample_load:
                for k in range(K):
                    lam_K[k], mu_K[k] = sample_mu_lam(samples, nK, ks, k, _mu0, _ka0, _a0, _b0)

        # the last iter
        pass
        tf.summary.scalar('dmm_epoch/sse', error, step=global_epoch)
        tf.summary.scalar('dmm_epoch/acc', acc, step=global_epoch)
        tf.summary.scalar('dmm_epoch/v_score', v_score, step=global_epoch)
        tf.summary.scalar('dmm_epoch/adj_rand', adj_rand, step=global_epoch)
        tf.summary.scalar('dmm_epoch/pair_f1', pair_f1, step=global_epoch)
        tf.summary.scalar('dmm_epoch/nmi', nmi, step=global_epoch)
        tf.summary.scalar('dmm_epoch/w_entropy', w_ent, step=global_epoch)
    # update params
    dir_params.K, dir_params.lam_K, dir_params.mu_K, dir_params.n_K, dir_params.samples_k = K, lam_K, mu_K, nK, ks
    return dir_params


# 每轮从0开始，每轮延续，每轮刷新？
def update_cluster_center_by_repr(samples_flow_z, dir_params):
    """
    after each epoch, repr is updated by flow, but the mean & lam are calced by old repr.
    maybe  we need to `refresh` them with new repr
    :param samples_flow_z: 
    :param dir_params: 
    :return: 
    """
    hp = dir_params.hyper
    _mu0, _ka0, logalpha, _a0, _b0 = hp.mu0, hp.ka0, hp.logalpha, hp.a0, hp.b0
    K, lam_K, mu_K, nK, ks = dir_params.K, dir_params.lam_K, dir_params.mu_K, dir_params.n_K, dir_params.samples_k
    for k in range(K):
        lam_K[k], mu_K[k] = sample_mu_lam(samples_flow_z, nK, ks, k, _mu0, _ka0, _a0, _b0)
    dir_params.lam_K, dir_params.mu_K, = lam_K, mu_K
    return dir_params


def train_flow(global_epoch, tb_writter, model_nice, opt, ds_inf_iter, n_iter, dir_params, args,
               log_step=0, prev_z_repr=None):
    if n_iter == 0:
        print('flow_opt=0, train ignored')
        return
    model_nice = model_nice.to(args.device)
    epoch_n_batch = args.n_sample_load // TRAIN_BATCH_SIZE
    epoch_loss, epoch = 0.0, 0
    K, ks, ns, mu_K, lam_K = dir_params.K, dir_params.samples_k, dir_params.n_K, dir_params.mu_K, dir_params.lam_K

    model_nice.to(args.device)
    with tb_writter.as_default():
        for n_batch in range(n_iter):
            (_, x, _, idx) = next(ds_inf_iter)
            x = x.to(args.device)

            # 1. calc loss
            model_nice.train()
            opt.zero_grad()
            # x = x.view(-1, args.dim)  # + (torch.rand(784)-0.5) / 256.
            x = x.to(args.device)
            _, likelihood = model_nice(x, K, ks[idx.numpy()], ns, np.asarray(mu_K), np.asarray(lam_K), args.reglam,
                                       args.contralam, args.hinge)
            if isinstance(likelihood, int):
                continue
            loss = -torch.mean(likelihood)  # NLL

            # 2. report real loss
            tf.summary.scalar('flow/batch_nll_loss', loss.detach().cpu().numpy(), step=n_batch + global_epoch * n_iter)
            if (n_batch + 1) % epoch_n_batch == 0:
                tb_writter.flush()
                tf.summary.scalar('flow/epoch_nll_loss', epoch_loss / epoch_n_batch,
                                  step=int(epoch + global_epoch * (n_iter / epoch_n_batch)))
                epoch_loss *= 0.0
                epoch += 1
            else:
                epoch_loss += loss.detach().cpu().numpy()

            # 3. cliping and applying gradients
            if args.loss_clip > 0:
                loss = torch.clip(loss, max=args.loss_clip)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_nice.parameters(), args.grad_clip)
            opt.step()

            # refresh cluster center
            if (n_batch + 1) % epoch_n_batch == 0:
                if args.refresh_cc_in_flow == 'yes':
                    dir_params = update_cluster_center_by_repr(prev_z_repr, dir_params)
                    K, ks, ns, mu_K, lam_K = dir_params.K, dir_params.samples_k, dir_params.n_K, dir_params.mu_K, dir_params.lam_K


def init_dir_params(args):
    dir_params = Namespace(
        hyper=Namespace(
            a0=args.a0 * args.dim, b0=args.b0 * args.dim,
            mu0=np.zeros(args.dim), ka0=args.kappa0,
            logalpha=args.logalpha
        ),
        K=0, n_K=[], lam_K=[], mu_K=[], samples_k=np.ones(args.n_sample_load, dtype=int) * -1
        # K: total num of clusters. lam_K, mu_K: sampled mu/lambda cluster. n_K: size of clusters
        # samples_k: cluster index for each sample, -1 means unassigned
    )
    return dir_params


def init_flow_model(args):
    model_nice = NICE(data_dim=args.dim, num_coupling_layers=args.nice_nlayers,
                      num_hidden_units=args.nice_units, device_name=args.device)
    opt = optim.Adam(model_nice.parameters(), args.lr)
    return model_nice, opt


def load_ae_dataset(ds_name, n_sample_load, aex_file):
    print('start loading ae dataset')
    if ds_name == 'mnist':
        ds_raw = load_dataset(ds_name, './data', split='train')
        ds_ae_repr = torch.load(aex_file)[:n_sample_load]
        ds_ae = AEDataset(ds_raw, ds_ae_repr, n_sample_load)
    elif 'VaDE_' in ds_name:
        X, Y, Z = np.load(f'../VaDE/{ds_name}_full_X.npz')['arr_0'], \
                  np.load(f'../VaDE/{ds_name}_full_Y.npz')['arr_0'].reshape(-1, ), \
                  np.load(f'../VaDE/{ds_name}_full_Z.npz')['arr_0']
        # X = (X - X.mean()) / X.std()  # todo critical norm
        ds_ae_repr = torch.from_numpy(Z)[:n_sample_load]
        ds_ae = OtherAEDataset(X, Y, Z, n_sample_load)
    else:
        raise ValueError(f'dataset name {ds_name} not supported')
    return ds_ae_repr, ds_ae


def main(manual_args=None):
    args = parse_args(manual_args)
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_id = '%s_%s_N%d_E%d_kappa0_%s_a0_%s_b0_%s_alpha_%s_%s_p%s' % (
            args.exp_name, args.dataset, args.n_sample_load, args.epoch, args.kappa0, args.a0, args.b0, args.logalpha, time_str, os.getpid())
    task_dir = os.path.join(args.log_dir, exp_id)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # 1. load AE dataset
    ds_ae_repr, ds_ae = load_ae_dataset(args.dataset, args.n_sample_load, args.aex_file)
    dl_ae = InfiniteDataLoader(dataset=ds_ae, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)
    iter_ae_inf = iter(dl_ae)
    print(f'data loaded, cls dist: N={args.n_sample_load}'
          f'{np.stack(np.unique(ds_ae.targets, return_counts=True))}')

    # 2. init model nice
    model_nice, opt = init_flow_model(args)

    # 3. init dirichlet mixture model
    dir_params = init_dir_params(args)

    # 4.alternated training
    tb_logger = tf.summary.create_file_writer(task_dir)
    with tb_logger.as_default():
        tf.summary.text('config', str(args), step=0)
        tf.summary.text('flow arch', str(model_nice), step=0)
        tf.summary.histogram('total distance norm init AE', torch.norm(ds_ae_repr, dim=1).cpu().numpy(), step=-1)
        tb_logger.flush()
    global_step_dmm, global_step_flow = 0, 0
    for n_epoch in range(args.epoch):
        print(f'{os.getpid()}-DDPM training epoch {n_epoch}')
        # 1. update sample representation
        # to_use_flow = None if args.with_flow == 'no' or n_epoch == 0 else model_nice
        # 第一轮用不用flow的影响不大 (reuters10k上简单测验了下)
#         samples_flow_z = get_flow_repr(ds_ae_repr, model_nice, args.device, normalize=args.input_normalize)
        samples_flow_z = transform_z(model_nice, ds_ae, args.n_sample_load)
#         samples_flow_z = samples_flow_z.detach().cpu().numpy()
        with tb_logger.as_default():
            distance = np.linalg.norm(samples_flow_z, axis=1)
            tf.summary.histogram('total distance norm flow', distance, step=n_epoch)

            distance_normed = np.linalg.norm((samples_flow_z-samples_flow_z.mean(axis=0)) / samples_flow_z.std(axis=0), axis=1)
            tf.summary.histogram('total distance norm flow (normed)', distance_normed, step=n_epoch)
            tb_logger.flush()
        if args.refresh_cc_in_flow == 'yes' and n_epoch >= 1:  # refresh cluster center repr with updated samples repr
            update_cluster_center_by_repr(samples_flow_z, dir_params)

        # 2. dirichlet clustering
#         if args.restart_dmm == 'yes':
#             dir_params = init_dir_params(args)
        n_iter_clst = get_iter_num(n_epoch, args.n_sample_load, args.iter_dmm)
        dir_params = dirichlet_clustering(n_epoch, tb_logger, dir_params, samples_flow_z, ds_ae.targets, n_iter_clst, args, log_step=global_step_dmm)
        global_step_dmm += n_iter_clst+1

        # 3. flow model trainging
        if args.with_flow == 'yes':
            if args.restart_flow == 'yes':
                model_nice, opt = init_flow_model(args)
            n_iter_opt = int(np.ceil(get_iter_num(n_epoch, args.n_sample_load, args.iter_nice) / TRAIN_BATCH_SIZE))
            train_flow(n_epoch, tb_logger, model_nice, opt, iter_ae_inf, n_iter_opt, dir_params, args,
                       log_step=global_step_flow, prev_z_repr=samples_flow_z)
            global_step_flow += n_iter_opt+1
        else:
            print('without flow opt')

        # save params
        if (n_epoch+1) % args.save_freq == 0 or n_epoch == args.epoch-1:
            # save flow model
            torch.save(model_nice.state_dict(), os.path.join(task_dir, f'flow_E{n_epoch}.pt'))

            # save dpm params
            np.savez(os.path.join(task_dir, f'dpm_E{n_epoch}'), K=dir_params.K, ks=dir_params.samples_k, ns=dir_params.n_K,
                     mu_K=np.asarray(dir_params.mu_K), lam_K=np.asarray(dir_params.lam_K))


if __name__ == '__main__':
    main()
