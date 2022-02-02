#%%
import pickle
from typing import Tuple
import numpy as np
import pandas as pd
import os
import dgl
import torch
from tqdm import tqdm

class EsunGraphBuilder:
    def __init__(self, path: str, drop_edges_by = 0, g_type = 'homo', neg_edges_ratio = 1, use_cache = True):
        """init

        Args:
            path (str, optional): folder path of esun .csv files. Defaults to './esun'.
            drop (int, optional): drop ratio, higher for faster experiment. Defaults to 0.
        """
        self.__path = path
        self.__drop_edges_by = drop_edges_by
        self.__g_type = g_type
        self.__neg_edges_ratio = neg_edges_ratio
        self.__config = {'path': path, 'drop_edges_by': drop_edges_by, 'g_type': g_type, 'neg_edges_ratio': neg_edges_ratio}
        self.__use_cache = use_cache
        
        self.__cache_path = os.path.join(path, '.esungbuilder')
        if not os.path.exists(self.__cache_path):
            os.mkdir(self.__cache_path)
        else:
            with open(os.path.join(self.__cache_path, 'config.pkl'), 'rb') as f:
                config = pickle.load(f)
            if config != self.__config:
                self.__use_cache = False
            
        with open(os.path.join(self.__cache_path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.__config, f)
        
        #TODO: try other id assign 
        self.__to_new_id_mapping_dict = {}
        self.__to_old_id_mapping_dict = {}
        self.__num_of_active_users = 0
        self.__num_of_active_items = 0
        self.__num_nodes = 0 #self.__customer_df.shape[0] + self.__product_df.shape[0]
        self.__interactions_df = pd.DataFrame()
        self.interactions_df
        
        self.__customer_df = pd.DataFrame()
        self.__product_df = pd.DataFrame()
        self.customer_df
        self.product_df
        
        self.__pos_g = None
        self.__pos_g_train = None
        self.__pos_g_test = None
        self.__neg_g_train = None
        self.__neg_g_test = None
        self.__node_features = None
 
    def __to_consecutive_id(self, series: pd.Series, start_from = 0) -> Tuple[pd.Series, dict]:
        id_list = list(series.unique())
        map_dict = {}
        id_tmp = start_from
        for id in id_list:
            map_dict[id] = id_tmp
            id_tmp += 1
        return series.map(map_dict), map_dict

    def __cat_to_one_hot(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        value_list = df[col_name].unique().tolist()
        for value in value_list:
            if str(value) == 'nan': continue
            df[col_name + ':' + str(value)] = np.where(df[col_name] == value, 1, 0)
        df.drop(col_name, axis=1, inplace=True)
        return df
    
    @property
    def interactions_df(self) -> pd.DataFrame:
        if self.__interactions_df.empty:
            # read transactions csv, concat
            transactions_train_df = pd.read_csv(os.path.join(self.__path, 'interaction_train.csv'))
            interactions_train_df = transactions_train_df.groupby(['cust_no', 'wm_prod_code']).first()['txn_amt'].reset_index()
            interactions_train_df['eval_mask'] = False
            transactions_eval_df = pd.read_csv(os.path.join(self.__path, 'interaction_eval.csv'))
            interactions_eval_df = transactions_eval_df.groupby(['cust_no', 'wm_prod_code']).first()['txn_amt'].reset_index()
            interactions_eval_df['eval_mask'] = True
            interaction_df = pd.concat([interactions_train_df, interactions_eval_df])

            # drop interaction for faster experiment
            use_propotion = 1 - self.__drop_edges_by
            if use_propotion < 1:
                use_cust_no = pd.Series(interaction_df['cust_no'].unique()).sample(frac=use_propotion).to_list()
                interaction_df = interaction_df[interaction_df['cust_no'].isin(use_cust_no)]
                return
            else:
                interaction_df = interaction_df
            
            # to consecutive id (reserve most feature)
            # interaction_df['user_id'] = interaction_df['cust_no'].map(self.to_new_id_mapping_dict)
            # interaction_df['item_id'] = interaction_df['wm_prod_code'].map(self.to_new_id_mapping_dict)
            
            # to consecutive id (reserve most interactions)
            interaction_df['user_id'], cust_no_to_user_id = self.__to_consecutive_id(interaction_df['cust_no'])
            self.__num_of_active_users = len(cust_no_to_user_id)
            interaction_df['item_id'], wm_prod_code_to_item_id = self.__to_consecutive_id(interaction_df['wm_prod_code'], start_from=self.__num_of_active_users)
            self.__to_new_id_mapping_dict.update(cust_no_to_user_id)
            self.__to_new_id_mapping_dict.update(wm_prod_code_to_item_id)
            self.__num_nodes = len(self.to_new_id_mapping_dict)
            
            interaction_df.drop(interaction_df.columns.difference(['user_id', 'item_id', 'eval_mask']), axis=1, inplace=True)
            
            # interaction_df.dropna(inplace=True)

            self.__interactions_df = interaction_df
        
        return self.__interactions_df
    
    @property
    def num_of_active_users(self):
        return self.__num_of_active_users

    @property
    def num_nodes(self):
        return self.__num_nodes
    
    @property
    def customer_df(self) -> pd.DataFrame:
        """load customer 1-hot features dataframe

        Args:
            path (str): folder path of esun .csv files
            drop_redundent (bool, optional): whether drop the users that never appear in the interaction table.

        Returns:
            pd.DataFrame: each row = [user_id, feat1, feat2, ...]
        """
        if self.__customer_df.empty:
            customer_df = pd.read_csv(os.path.join(self.__path, 'customer.csv'))
            customer_df.drop(customer_df.columns.difference(['cust_no', 'gender_code', 'age', 'income_range_code', 'risk_type_code']), axis=1, inplace=True)

            customer_df = self.__cat_to_one_hot(customer_df, 'gender_code')

            customer_df['age:1'] = np.where(customer_df['age'] < 18, 1, 0)
            customer_df['age:18'] = np.where((customer_df['age'] >= 18) & (customer_df['age'] < 25), 1, 0)
            customer_df['age:25'] = np.where((customer_df['age'] >= 25) & (customer_df['age'] < 35), 1, 0)
            customer_df['age:35'] = np.where((customer_df['age'] >= 35) & (customer_df['age'] < 45), 1, 0)
            customer_df['age:45'] = np.where((customer_df['age'] >= 45) & (customer_df['age'] < 50), 1, 0)
            customer_df['age:50'] = np.where((customer_df['age'] >= 50) & (customer_df['age'] < 56), 1, 0)
            customer_df['age:56'] = np.where(customer_df['age'] >= 56, 1, 0)
            customer_df.drop('age', axis=1, inplace=True)

            customer_df = self.__cat_to_one_hot(customer_df, 'income_range_code')
            customer_df = self.__cat_to_one_hot(customer_df, 'risk_type_code')

            # customer_df['user_id'], cust_no_to_user_id = self.__to_consecutive_id(customer_df['cust_no'])
            # self.__to_new_id_mapping_dict.update(cust_no_to_user_id)
            customer_df['user_id'] = customer_df['cust_no'].map(self.to_new_id_mapping_dict)
            customer_df.drop('cust_no', axis=1, inplace=True)
            
            self.__customer_df = customer_df
        
        return self.__customer_df

    @property
    def product_df(self) -> pd.DataFrame:
        if self.__product_df.empty:
            product_df = pd.read_csv(os.path.join(self.__path, 'product.csv'))
            product_df.drop(['prod_type_code', 'Unnamed: 0'], axis=1, inplace=True)

            product_df = self.__cat_to_one_hot(product_df, 'high_yield_bond_ind')
            product_df = self.__cat_to_one_hot(product_df, 'invest_type')
            product_df = self.__cat_to_one_hot(product_df, 'mkt_rbot_ctg_ic')
            product_df = self.__cat_to_one_hot(product_df, 'prod_detail_type_code')
            product_df = self.__cat_to_one_hot(product_df, 'prod_ccy')
            product_df = self.__cat_to_one_hot(product_df, 'prod_risk_code')
            product_df = self.__cat_to_one_hot(product_df, 'can_rcmd_ind')

            # product_df['item_id'], wm_prod_code_to_item_id = self.__to_consecutive_id(product_df['wm_prod_code'], start_from=self.customer_df.shape[0])
            # self.to_new_id_mapping_dict.update(wm_prod_code_to_item_id)
            product_df['item_id'] = product_df['wm_prod_code'].map(self.to_new_id_mapping_dict)
            product_df.drop('wm_prod_code', axis=1, inplace=True)
            
            self.__product_df = product_df
            
        return self.__product_df
    
    @property
    def to_new_id_mapping_dict(self) -> dict:
        if self.__to_new_id_mapping_dict == {}:
            self.interactions_df
        
        return self.__to_new_id_mapping_dict

    @property
    def to_old_id_mapping_dict(self) -> dict:
        if self.__to_old_id_mapping_dict == {}:
            self.__to_old_id_mapping_dict = dict((v,k) for k,v in self.to_new_id_mapping_dict.items())
        
        return self.__to_old_id_mapping_dict      

    def __get_k_negative_edges_from_g(self, g: dgl.DGLHeteroGraph, k = 1):
        u_list, v_list = g.edges()
        u_list, v_list = list(u_list), list(v_list)
        pos_edges_df = pd.DataFrame({'u':u_list, 'v':v_list})
        neg_edges_df = pd.DataFrame({'u':[], 'v':[]})
        
        # print('start to find ' + str(k) + ' edges...')
        with tqdm(total=k, desc='Sample neg. edges : ') as pbar:
            num_edges = 0
            while neg_edges_df.shape[0] < k:
                neg_u_list, neg_v_list = np.random.choice(u_list, k), np.random.choice(v_list, k)
                edges_df = pd.DataFrame({'u':neg_u_list, 'v':neg_v_list})
                edges_df = edges_df[edges_df['u'] != edges_df['v']]
                neg_edges_df = neg_edges_df.append(edges_df, ignore_index=True)
                neg_edges_df.drop_duplicates(inplace=True)
                neg_edges_df = neg_edges_df[~neg_edges_df.isin(pos_edges_df)].dropna()
                pbar.update((neg_edges_df.shape[0] - num_edges))
                num_edges = neg_edges_df.shape[0]
                # if neg_edges_df.shape[0] > k * 0.9: break
                
            neg_edges_df = neg_edges_df[:k]
            pbar.update((neg_edges_df.shape[0] - num_edges))
        return neg_edges_df

    def __build_heter_graphs(self):
        #TODO: try undirect bipartite heter graph
        pass
    
    def __build_homo_graphs(self):
        g_list_path = os.path.join(self.__cache_path, 'graphs.dgl')
        
        if self.__use_cache and os.path.exists(g_list_path):
            g_list, tmp = dgl.load_graphs(g_list_path)
            self.__pos_g = g_list[0]
            self.__pos_g_train = g_list[1]
            self.__pos_g_test = g_list[2]
            self.__neg_g_train = g_list[3]
            self.__neg_g_test = g_list[4]
        else:            
            # build edge set for positive graphs
            pos_u = self.interactions_df['user_id'].to_list()
            pos_v = self.interactions_df['item_id'].to_list()
            pos_u_train = self.interactions_df[~self.interactions_df['eval_mask']]['user_id'].to_list()
            pos_v_train = self.interactions_df[~self.interactions_df['eval_mask']]['item_id'].to_list()
            pos_u_test = self.interactions_df[self.interactions_df['eval_mask']]['user_id'].to_list()
            pos_v_test = self.interactions_df[self.interactions_df['eval_mask']]['item_id'].to_list()
            
            # build positive graph
            pos_g = dgl.graph((pos_u, pos_v), num_nodes=self.num_nodes)
            pos_g = dgl.to_simple(pos_g)
            pos_g = dgl.to_bidirected(pos_g)
            
            pos_g_train = dgl.graph((pos_u_train, pos_v_train), num_nodes=self.__num_nodes)
            pos_g_train = dgl.to_bidirected(pos_g_train)
            
            pos_g_test = dgl.graph((pos_u_test, pos_v_test), num_nodes=self.__num_nodes)
            pos_g_test = dgl.to_bidirected(pos_g_test)
            
            # find all negative edges and split them for trainnig and testing (undirect)
            # neg_edges = self.__get_k_negative_edges_from_g(pos_g, pos_g.num_edges() // 2)
            # neg_u_train = neg_edges['u'].to_list()[pos_g_test.num_edges():]
            # neg_v_train = neg_edges['v'].to_list()[pos_g_test.num_edges():]
            # neg_u_test = neg_edges['u'].to_list()[:pos_g_test.num_edges()]
            # neg_v_test = neg_edges['v'].to_list()[:pos_g_test.num_edges()]
            
            # neg_g_train = dgl.graph((neg_u_train, neg_v_train), num_nodes=self.__num_nodes)
            # neg_g_train = dgl.to_bidirected(neg_g_train)
            # neg_g_test = dgl.graph((neg_u_test, neg_v_test), num_nodes=self.__num_nodes)
            # neg_g_test = dgl.to_bidirected(neg_g_test)
            
            # find all negative edges and split them for trainnig and testing (direct)
            neg_edges = self.__get_k_negative_edges_from_g(pos_g, int(pos_g.num_edges() * self.__neg_edges_ratio))
            neg_u_train = neg_edges['u'].to_list()[pos_g_test.num_edges():]
            neg_v_train = neg_edges['v'].to_list()[pos_g_test.num_edges():]
            neg_u_test = neg_edges['u'].to_list()[:pos_g_test.num_edges()]
            neg_v_test = neg_edges['v'].to_list()[:pos_g_test.num_edges()]
            
            neg_g_train = dgl.graph((neg_u_train, neg_v_train), num_nodes=self.__num_nodes)
            # neg_g_train = dgl.to_bidirected(neg_g_train)
            neg_g_test = dgl.graph((neg_u_test, neg_v_test), num_nodes=self.__num_nodes)
            # neg_g_test = dgl.to_bidirected(neg_g_test)
            

            self.__pos_g = pos_g
            self.__pos_g_train = pos_g_train
            self.__pos_g_test = pos_g_test
            self.__neg_g_train = neg_g_train
            self.__neg_g_test = neg_g_test

            dgl.save_graphs(g_list_path, [pos_g, pos_g_train, pos_g_test, neg_g_train, neg_g_test])
    
    def build_graphs(self):
        if self.__g_type == 'homo':
            self.__build_homo_graphs()
        elif self.__g_type == 'heter':
            self.__build_heter_graphs()
            
        self.node_features
        # self.pos_g.ndata['feat'] = self.node_features
        # self.pos_g_train.ndata['feat'] = self.node_features
    
    @property
    def pos_g(self) -> dgl.DGLHeteroGraph:
        if self.__pos_g == None:
            self.build_graphs()
        return self.__pos_g
    
    @property
    def pos_g_train(self) -> dgl.DGLHeteroGraph:
        if self.__pos_g_train == None:
            self.build_graphs()
        return self.__pos_g_train
    
    @property
    def pos_g_test(self) -> dgl.DGLHeteroGraph:
        if self.__pos_g_test == None:
            self.build_graphs()
        return self.__pos_g_test
    
    @property
    def neg_g_train(self) -> dgl.DGLHeteroGraph:
        if self.__neg_g_train == None:
            self.build_graphs()
        return self.__neg_g_train
    
    @property
    def neg_g_test(self) -> dgl.DGLHeteroGraph:
        if self.__neg_g_test == None:
            self.build_graphs()
        return self.__neg_g_test
    
    def __build_homo_nodes_feature(self):
        if self.__node_features == None:
            node_features_path = os.path.join(self.__cache_path, 'node_feat.pt')
            if self.__use_cache and os.path.exists(node_features_path):
                self.__node_features = torch.load(node_features_path)
            else:
                # assign feature to pos graph
                feat_list = []
                with tqdm(total=self.__pos_g.num_nodes(), desc='Assign feature : ') as pbar:
                    for node_id in range(self.__pos_g.num_nodes()):
                        customer_feat_size = len(self.customer_df.columns) - 1
                        product_feat_size = len(self.product_df.columns) - 1
                        if node_id < self.__num_of_active_users: # assign user node features
                            try:
                                feat = self.customer_df[self.customer_df['user_id'] == node_id].values[0][:-1]
                                feat = torch.cat((torch.tensor(feat), torch.zeros(product_feat_size)))
                            except:
                                feat = torch.zeros(customer_feat_size + product_feat_size)
                        else:  # assign item node features
                            try:
                                feat = self.product_df[self.product_df['item_id'] == node_id].values[0][:-1]
                                # feat = torch.cat((torch.zeros(customer_feat_size), torch.tensor(feat))) 
                                feat = torch.cat((torch.tensor(feat), torch.zeros(customer_feat_size))) #TODO: why other better?
                            except:
                                feat = torch.zeros(customer_feat_size + product_feat_size)
                        feat_list.append(feat)
                        pbar.update()
                feat_list = torch.stack(feat_list)
                
                torch.save(feat_list, node_features_path)
                self.__node_features = feat_list
        
    @property
    def node_features(self) -> torch.Tensor:
        if self.__node_features == None:
            node_features_path = os.path.join(self.__cache_path, 'node_feat.pt')
            if self.__use_cache and os.path.exists(node_features_path):
                self.__node_features = torch.load(node_features_path)
            else:
                if self.__g_type == 'homo':
                    self.__build_homo_nodes_feature()
        
        return self.__node_features
        
def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    esun_csv_folder_path = os.path.join(this_dir_path, 'esun')

    esun = EsunGraphBuilder(esun_csv_folder_path, use_cache=False)
    esun.build_graphs()

# print(esun.pos_g)
# print(esun.node_features)

# if __name__ == '__main__':
#     main()
# %%
