#%%
import argparse
from statistics import mode
from importlib_metadata import itertools
import numpy as np
from data_utils import EsunGraphBuilder
from model import GraphSAGE, MLPPredictor, compute_auc, compute_loss
from reco_utils import Evaluation
from tqdm import tqdm
import dgl
import pandas as pd
import bottleneck as bn
import os
import random
import wandb
from datetime import datetime, timedelta
import torch

enable_wandb = False

if enable_wandb:
    wandb.init(project="link_predict_esun_v0", tags=['test_0208'])

#%%
# setup config

# python ./link_predict.py --dataset_path esun --neg-edges-ratio 2 --num-epochs 2000 --hidden-dims 32 --agg-type mean --reco-ratio 0.1 --use-cache True

try:    
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--neg-edges-ratio', type=float, default=1)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--agg-type', type=str, default='mean')
    parser.add_argument('--reco-ratio', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use-cache', type=bool, default=True)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    neg_edges_ratio = args.neg_edges_ratio
    hidden_dims = args.hidden_dims
    agg_type = args.agg_type
    num_epochs = args.num_epochs
    reco_ratio = args.reco_ratio
    lr = args.lr
    use_cache = args.use_cache
    
    print('\nuse command line\n')
except:
    dataset_path = 'esun'
    neg_edges_ratio = 2
    hidden_dims = 64
    agg_type = 'mean'
    num_epochs = 6000
    reco_ratio = 1
    lr = 0.00125
    use_cache = False
    
    print('\nuse jupyter\n')

if enable_wandb:
    wandb.config['dataset_path'] = dataset_path
    wandb.config['neg_edges_ratio'] = neg_edges_ratio
    wandb.config['hidden_dims'] = hidden_dims
    wandb.config['agg_type'] = agg_type
    wandb.config['num_epochs'] = num_epochs
    wandb.config['reco_ratio'] = reco_ratio
    wandb.config['lr'] = lr
    wandb.config['use_cache'] = use_cache

#%%
# load esun graph
esun = EsunGraphBuilder(dataset_path, neg_edges_ratio=neg_edges_ratio, use_cache=use_cache, g_type='homo')
esun.build_graphs()

#%%
# instantiate model, predictor, optimizer (homo)
model = GraphSAGE(esun.node_features.shape[1], hidden_dims, agg_type=agg_type)
pred = MLPPredictor(hidden_dims)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=lr)

#%%
# to cuda/cpu
all_logits = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ' + str(device))
model = model.to(device)
pred = pred.to(device)
pos_g_train = esun.pos_g_train.to(device)
neg_g_train = esun.neg_g_train.to(device)
pos_g_test = esun.pos_g_test.to(device)
neg_g_test = esun.neg_g_test.to(device)
feat = esun.node_features.to(device)

#%%
# train
loss_train_log = []
auc_train_log = []
best_h = None
model = model.float()
with tqdm(total=num_epochs, desc='Train : ') as pbar:
    min_loss = 100000
    max_auc = 0
    for e in range(num_epochs):
        # forward
        h = model(pos_g_train, feat.float())
        pos_score = pred(pos_g_train, h)
        neg_score = pred(neg_g_train, h)
        loss = compute_loss(pos_score, neg_score)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # check auc
        if e % 50 == 0:
            with torch.no_grad():
                pos_score = pred(pos_g_test, h)
                neg_score = pred(neg_g_test, h)
                auc = compute_auc(pos_score, neg_score)
                if auc > max_auc:
                    max_auc = auc

        # feedback and record
        pbar.update(1)
        pbar.set_postfix_str('loss : ' + str(round(float(loss), 4)) + ', auc : ' + str(round(float(auc), 4)) + ', min_loss : ' + str(round(float(min_loss), 4)) + ', max_auc : ' + str(round(float(max_auc), 4)))

        loss_train_log.append(float(loss))
        auc_train_log.append(float(auc))

        # pick best
        if loss < min_loss:
            min_loss = loss
            
        if enable_wandb:
            wandb.log({'loss': loss, 'auc': auc, 'min_loss': min_loss, 'max_auc': max_auc})

#%%
# test
with torch.no_grad():
    pos_score = pred(pos_g_test, h)
    neg_score = pred(neg_g_test, h)
    print('AUC', compute_auc(pos_score, neg_score))
#%%
# recommend
def recommend_user(user_id, top_k, node_feat, pred):
    pass

# %%
# recommend (all)
recommendation = {}

num_reco_user = int(esun.customer_df.shape[0] * reco_ratio)
with torch.no_grad():
    with tqdm(total=num_reco_user, desc='Recommend : ') as pbar:
        u_list = random.sample(range(esun.num_of_active_users), num_reco_user)
        v_list = list(range(esun.num_of_active_users, esun.num_nodes))
        
        for u in u_list:
            dense_g_u = []
            dense_g_v = []
            
            for v in v_list:
                dense_g_u.append(u)
                dense_g_v.append(v)

            dense_g = dgl.graph((dense_g_u, dense_g_v), num_nodes=esun.num_nodes)
            dense_g = dense_g.to(device)
            edata = pred(dense_g, h).cpu()

            user_reco = bn.argpartition(-edata, 5)[:5]
            for i in range(len(user_reco)):
                user_reco[i] = v_list[user_reco[i]]
            
            user_reco = list(map(lambda x: esun.to_old_id_mapping_dict[x], user_reco))
            recommendation[esun.to_old_id_mapping_dict[u]] = user_reco

            pbar.update()

# %%
# evaluate (all)
print('evaluating...')
evaluation = Evaluation('', './esun/interaction_eval.csv', recommendation)
score = evaluation.results()
print("===",score)

# %%
# recommend (new item)
recommendation_new_item = {}

num_reco_user = int(esun.customer_df.shape[0] * reco_ratio)
with torch.no_grad():
    with tqdm(total=num_reco_user, desc='Recommend : ') as pbar:
        u_list = random.sample(range(esun.num_of_active_users), num_reco_user)
        v_list = list(range(esun.num_of_active_users, esun.num_nodes))
        
        v_exclude_list_for_each_u = {}
        observe_date = datetime(2019, 6, 30)
        interaction_df_train_recent = esun.interactions_df
        interaction_df_train_recent['txn_dt'] = pd.to_datetime(interaction_df_train_recent['txn_dt'])
        interaction_df_train_recent = interaction_df_train_recent[(observe_date - interaction_df_train_recent['txn_dt']) < timedelta(days=90)]
        interaction_df_train_recent = interaction_df_train_recent[~interaction_df_train_recent['eval_mask']]
        v_exclude_list_for_each_u = interaction_df_train_recent.groupby('user_id')['item_id'].apply(list).to_dict()
        
        for u in u_list:
            dense_g_u = []
            dense_g_v = []
            try:
                v_exclude_list = v_exclude_list_for_each_u[u]
            except:
                v_exclude_list = []
            
            for v in v_list:
                if not v in v_exclude_list:
                    dense_g_u.append(u)
                    dense_g_v.append(v)

            dense_g = dgl.graph((dense_g_u, dense_g_v), num_nodes=esun.num_nodes)
            dense_g = dense_g.to(device)
            edata = pred(dense_g, h).cpu()

            user_reco = bn.argpartition(-edata, 5)[:5]
            for i in range(len(user_reco)):
                user_reco[i] = v_list[user_reco[i]]
            
            user_reco = list(map(lambda x: esun.to_old_id_mapping_dict[x], user_reco))
            recommendation_new_item[esun.to_old_id_mapping_dict[u]] = user_reco

            pbar.update()

# %%
# evaluate (new item)
print('evaluating...')
evaluation = Evaluation('', './esun/interaction_eval.csv', recommendation_new_item)
score_new_item = evaluation.results()
print("===",score_new_item)

# %%
# recommend (upper))
recommendation_upper = {}

num_reco_user = int(esun.customer_df.shape[0] * reco_ratio)
with torch.no_grad():
    with tqdm(total=num_reco_user, desc='Recommend : ') as pbar:
        u_list = random.sample(range(esun.num_of_active_users), num_reco_user)
        v_list = list(range(esun.num_of_active_users, esun.num_nodes))
        
        for u in u_list:
            # dense_g_u = []
            # dense_g_v = []
            
            # for v in v_list:
            #     dense_g_u.append(u)
            #     dense_g_v.append(v)

            # dense_g = dgl.graph((dense_g_u, dense_g_v), num_nodes=esun.num_nodes)
            # dense_g = dense_g.to(device)
            # edata = pred(dense_g, h).cpu()

            # user_reco = bn.argpartition(-edata, 5)[:5]
            # for i in range(len(user_reco)):
            #     user_reco[i] = v_list[user_reco[i]]

            user_reco = v_list
            user_reco = list(map(lambda x: esun.to_old_id_mapping_dict[x], user_reco))
            recommendation_upper[esun.to_old_id_mapping_dict[u]] = user_reco

            pbar.update()

# %%
# evaluate (upper)
print('evaluating...')
evaluation = Evaluation('', './esun/interaction_eval.csv', recommendation_upper)
score_upper = evaluation.results()
print("===",score_upper)

# %%
# recommend (upper of new)
recommendation_upper_new = {}

num_reco_user = int(esun.customer_df.shape[0] * reco_ratio)
with torch.no_grad():
    with tqdm(total=num_reco_user, desc='Recommend : ') as pbar:
        u_list = random.sample(range(esun.num_of_active_users), num_reco_user)
        v_list = list(range(esun.num_of_active_users, esun.num_nodes))
        
        v_exclude_list_for_each_u = {}
        observe_date = datetime(2019, 6, 30)
        interaction_df_train_recent = esun.interactions_df
        interaction_df_train_recent['txn_dt'] = pd.to_datetime(interaction_df_train_recent['txn_dt'])
        interaction_df_train_recent = interaction_df_train_recent[(observe_date - interaction_df_train_recent['txn_dt']) < timedelta(days=90)]
        interaction_df_train_recent = interaction_df_train_recent[~interaction_df_train_recent['eval_mask']]
        v_exclude_list_for_each_u = interaction_df_train_recent.groupby('user_id')['item_id'].apply(list).to_dict()
        
        for u in u_list:
            # dense_g_u = []
            # dense_g_v = []
            # try:
            #     v_exclude_list = v_exclude_list_for_each_u[u]
            # except:
            #     v_exclude_list = []
            
            # for v in v_list:
            #     if not v in v_exclude_list:
            #         dense_g_u.append(u)
            #         dense_g_v.append(v)

            # dense_g = dgl.graph((dense_g_u, dense_g_v), num_nodes=esun.num_nodes)
            # dense_g = dense_g.to(device)
            # edata = pred(dense_g, h).cpu()

            # user_reco = bn.argpartition(-edata, 5)[:5]
            # for i in range(len(user_reco)):
            #     user_reco[i] = v_list[user_reco[i]]

            user_reco = v_list
            try:
                v_exclude_list = v_exclude_list_for_each_u[u]
            except:
                v_exclude_list = []
            user_reco = list(set(v_list) - set(v_exclude_list))

            user_reco = list(map(lambda x: esun.to_old_id_mapping_dict[x], user_reco))
            recommendation_upper_new[esun.to_old_id_mapping_dict[u]] = user_reco

            pbar.update()

# %%
# evaluate (upper of new)
print('evaluating...')
evaluation = Evaluation('', './esun/interaction_eval.csv', recommendation_upper_new)
score_upper_new = evaluation.results()
print("===",score_upper_new)

#%%
# finish wandb run
if enable_wandb:
    wandb.run.summary['score'] = score
    wandb.run.summary['score_new_item'] = score_new_item
    wandb.run.summary['score_upper'] = score_upper
    wandb.run.summary['score_upper_new_item'] = score_upper_new
    wandb.finish()

# %%
