#%%
import argparse
from statistics import mode
from importlib_metadata import itertools
import numpy as np
from data_utils import EsunGraphBuilder
from model import *
from tqdm import tqdm
import dgl
import pandas as pd
import bottleneck as bn
import os

#%%
# setup config
parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str)
parser.add_argument('--neg-edges-ratio', type=float, default=1)
parser.add_argument('--num-epochs', type=int, default=5)
parser.add_argument('--hidden-dims', type=int, default=16)
parser.add_argument('--agg-type', type=str, default='mean')
parser.add_argument('--reco-ratio', type=float, default=1)
args = parser.parse_args()

dataset_path = args.dataset_path
neg_edges_ratio = args.neg_edges_ratio
hidden_dims = args.hidden_dims
agg_type = args.agg_type
num_epochs = args.num_epochs
reco_ratio = args.reco_ratio

# dataset_path = 'esun'
# neg_edges_ratio = 1
# hidden_dims = 64
# agg_type = 'mean'
# num_epochs = 3000
# reco_ratio = 0.1

#%%
# load esun graph
esun = EsunGraphBuilder(dataset_path, neg_edges_ratio=neg_edges_ratio, use_cache=True)
esun.build_graphs()

#%%
# instantiate model, predictor, optimizer
model = GraphSAGE(esun.node_features.shape[1], hidden_dims, agg_type=agg_type)
pred = MLPPredictor(hidden_dims)
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

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
                    best_h = h.clone().detach()
                # print('AUC', compute_auc(pos_score, neg_score))
        
        # feedback and record
        pbar.update(1)
        pbar.set_postfix_str('loss : ' + str(round(float(loss), 4)) + ', auc : ' + str(round(float(auc), 4)) + ', min_loss : ' + str(round(float(min_loss), 4)) + ', max_auc : ' + str(round(float(max_auc), 4)))
        
        loss_train_log.append(float(loss))
        auc_train_log.append(float(auc))

        # pick best
        if loss < min_loss:
            min_loss = loss
    # h = best_h.clone().detach()

#%%
# test
with torch.no_grad():
    pos_score = pred(pos_g_test, best_h)
    neg_score = pred(neg_g_test, best_h)
    print('AUC', compute_auc(pos_score, neg_score))

# %%
# recommend
recommendation = {}

num_reco_user = int(esun.customer_df.shape[0] * reco_ratio)
with torch.no_grad():
    with tqdm(total=num_reco_user, desc='Recommend : ') as pbar:
        u_list = np.random.choice(esun.num_of_active_users, num_reco_user)
        v_list = list(range(esun.num_of_active_users, esun.num_nodes))
        
        for u in u_list:
            dense_g_u = []
            dense_g_v = []
            for v in v_list:
                dense_g_u.append(u)
                dense_g_v.append(v)
                
            dense_g = dgl.graph((dense_g_u, dense_g_v), num_nodes=esun.node_features.shape[0])
            dense_g = dense_g.to(device)
            edata = pred(dense_g, best_h).cpu()

            user_reco = bn.argpartition(-edata, 5)[:5]
            for i in range(len(user_reco)):
                user_reco[i] = v_list[user_reco[i]]
            user_reco = list(map(lambda x: esun.to_old_id_mapping_dict[x], user_reco))
            recommendation[esun.to_old_id_mapping_dict[u]] = user_reco

            pbar.update()

# %%
# evaluate
print('evaluating...')
evaluation = Evaluation('', './esun/interaction_eval.csv', recommendation)
score = evaluation.results()
print("===",score)

#%%
# save logs
logs_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_logs')
if not os.path.exists(logs_folder_path):
    os.mkdir(logs_folder_path)
loss_path = os.path.join(logs_folder_path, 'loss.csv')
auc_path = os.path.join(logs_folder_path, 'auc.csv')
p_at_n_path = os.path.join(logs_folder_path, 'p_at_n.csv')

col_name = agg_type + '_h' + str(hidden_dims) + '_e' + str(num_epochs) + '_mlp'
if os.path.exists(loss_path) and os.path.exists(p_at_n_path):
    loss_df = pd.read_csv(loss_path)
    loss_df[col_name] = pd.Series(loss_train_log)
    loss_df.to_csv(loss_path, index=False)
    
    auc_df = pd.read_csv(auc_path)
    auc_df[col_name] = pd.Series(auc_train_log)
    auc_df.to_csv(auc_path, index=False)
    
    p_at_n_df = pd.read_csv(p_at_n_path)
    p_at_n_df[col_name] = pd.Series([score,])
    p_at_n_df.to_csv(p_at_n_path, index=False)
else:    
    loss_df = pd.DataFrame({col_name:loss_train_log})
    loss_df.to_csv(loss_path, index=False)
    
    auc_df = pd.DataFrame({col_name:auc_train_log})
    auc_df.to_csv(auc_path, index=False)
    
    p_at_n_df = pd.DataFrame({col_name:[score,]})
    p_at_n_df.to_csv(p_at_n_path, index=False)

# %%
