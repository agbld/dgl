from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import dgl.function as fn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(torch.device('cuda:0'))
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, agg_type = 'mean'):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, agg_type)
        self.conv2 = SAGEConv(h_feats, h_feats, agg_type)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
class Evaluation:
    
    def __init__(self, date, path, pred):
        self.today = date
        self.path = path
        self.pred = pred
        self.ans = self.answer(self.path)
    
    def show(self):
        print(f"Date: {self.today}\n") 
        coverage = len(set(self.pred.keys()) & set(self.ans.keys()))       
        print(f"Uppper-Bound: {coverage}\n")
 
    def answer(self, path):
        df = self.read(path)
        return df.groupby('cust_no')['wm_prod_code'].apply(list).to_dict()

    def read(self, path):
        return pd.read_csv(path, usecols=['cust_no', 'wm_prod_code'])  

    def results(self):
        p = 0
        count = len(self.ans)
        for u, pred in tqdm(self.pred.items(), total=count):
            p += self.precision_at_5(u, pred)
        return p/count
    
    def precision_at_5(self, user, pred):
        try:
            y_true = self.ans[user]
            tp = len(set(y_true) & set(pred))
            return tp/5
        except:
            return 0
        
