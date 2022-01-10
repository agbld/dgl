#%%
from os import sep
import os
from numpy import cos, int32
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str, default='./2018-12-31')
parser.add_argument('output_directory', type=str, default='./esun')
args = parser.parse_args()
directory = args.directory
output_directory = args.output_directory
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

def to_consecutive_id(series: pd.Series):
    id_list = list(series.values)
    map_dict = {}
    id_tmp = 1
    for id in id_list:
        map_dict[id] = id_tmp
        id_tmp += 1
    return series.map(map_dict), map_dict

#%%
# read esun data
customer_df = pd.read_csv(directory + '/customer.csv')
interaction_eval_df = pd.read_csv(directory + '/interaction_eval.csv')
interaction_train_df = pd.read_csv(directory + '/interaction_train.csv')
product_df = pd.read_csv(directory + '/product.csv')

#%%
# process customer
customer_df.drop(customer_df.columns.difference(['cust_no', 'gender_code', 'age', 'income_range_code', 'risk_type_code']), axis=1, inplace=True)
customer_df = customer_df[['cust_no', 'gender_code', 'age', 'income_range_code', 'risk_type_code']]

customer_df['cust_no'], cust_no_to_new = to_consecutive_id(customer_df['cust_no'])

customer_df['gender_code'] = customer_df['gender_code'].fillna('M')

customer_df['age'] = np.where(customer_df['age'] < 18, 1, customer_df['age'])
customer_df['age'] = np.where((customer_df['age'] >= 18) & (customer_df['age'] < 25), 18, customer_df['age'])
customer_df['age'] = np.where((customer_df['age'] >= 25) & (customer_df['age'] < 35), 25, customer_df['age'])
customer_df['age'] = np.where((customer_df['age'] >= 35) & (customer_df['age'] < 45), 35, customer_df['age'])
customer_df['age'] = np.where((customer_df['age'] >= 45) & (customer_df['age'] < 50), 45, customer_df['age'])
customer_df['age'] = np.where((customer_df['age'] >= 50) & (customer_df['age'] < 56), 50, customer_df['age'])
customer_df['age'] = np.where(customer_df['age'] >= 56, 56, customer_df['age'])
customer_df['age'] = customer_df['age'].astype(int32)

customer_df['income_range_code'] = customer_df['income_range_code'].fillna(0)
customer_df['income_range_code'] = customer_df['income_range_code'].astype(int32)

customer_df['risk_type_code'] = customer_df['risk_type_code'].fillna('00')
# risk_type_code_map_dict = {'00':0, '01':1, '02':2, '03':3, '04':4, 'ZZ':5}
# customer_df['risk_type_code'] = customer_df['risk_type_code'].map(risk_type_code_map_dict)

np.savetxt(output_directory + '/users.dat', customer_df.values, delimiter='::', fmt='%s',encoding='utf-8')

#%%
# process product
product_df['wm_prod_code'], wm_prod_code_to_new = to_consecutive_id(product_df['wm_prod_code'])
product_df['movie_id'] = product_df['wm_prod_code']
product_df['title'] = product_df['wm_prod_code'].astype(str) + '(2022)'
product_df['genres'] = product_df['invest_type'].astype(str) + '|' + product_df['prod_risk_code'].astype(str)
product_df.drop(product_df.columns.difference(['movie_id', 'title', 'genres']), axis=1, inplace=True)

np.savetxt(output_directory + '/movies.dat', product_df.values, delimiter='::', fmt='%s',encoding='utf-8')

#%%
# process interaction
interaction_train_df['user_id'] = interaction_train_df['cust_no'].map(cust_no_to_new)
interaction_train_df['movie_id'] = interaction_train_df['wm_prod_code'].map(wm_prod_code_to_new)
interaction_train_df['rating'] = 5
interaction_train_df['timestamp'] = interaction_train_df['txn_dt'].str.replace('-', '')
interaction_train_df.drop(interaction_train_df.columns.difference(['user_id', 'movie_id', 'rating', 'timestamp']), axis=1, inplace=True)
# print(interaction_train_df.shape[0])
interaction_train_df = interaction_train_df.dropna()
# print(interaction_train_df.shape[0])
interaction_train_df['user_id'] = interaction_train_df['user_id'].astype(int32)

np.savetxt(output_directory + '/ratings.dat', interaction_train_df.values, delimiter='::', fmt='%s',encoding='utf-8')
# %%