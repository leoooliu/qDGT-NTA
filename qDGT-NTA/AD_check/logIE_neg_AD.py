# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 09:23:12 2025

@author: 79234
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 23:31:58 2025

@author: 79234
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:13:26 2025

@author: 79234
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:12:00 2025

@author: 79234
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from rdkit.Avalon import pyAvalonTools
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from mordred import Calculator, descriptors
import numpy as np
from xgboost import XGBRegressor
from rdkit.Chem import  MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Avalon import pyAvalonTools
from sklearn.preprocessing import StandardScaler


# 初始化Mordred描述符计算器
calc = Calculator(descriptors, ignore_3D=True)

def generate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.array([])  # 返回一个空数组处理无效分子的情况

    # RDKit 描述符
    
 

    # Hashed Morgan 指纹
    hashed_morgan_fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=3, nBits=1024)
    hashed_morgan_array = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(hashed_morgan_fp, hashed_morgan_array)
    

    
  

    # Mordred 描述符
    mordred_desc = calc(mol)
    mordred_features = [x for x in mordred_desc.fill_missing().values()]


    
    # 融合特征
    features = np.concatenate([hashed_morgan_array, mordred_features])
   
  

    return features


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def generate_all_features(smiles_list):
    feature_list = []
    valid_index = []
    for i, smiles in enumerate(smiles_list):
        features = generate_features(smiles)
        if features is not None and len(features) > 0:
            feature_list.append(features)
            valid_index.append(i)
    return np.array(feature_list), valid_index

def check_applicability_domain(train_feat, new_feat, similarity_metric='cosine', similarity_threshold=0.55, count_threshold=3):
    if similarity_metric == 'cosine':
        similarities = cosine_similarity(train_feat, new_feat)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
    num_similar_train_samples = np.sum(similarities >= similarity_threshold, axis=0)
    in_domain_flags = num_similar_train_samples >= count_threshold
    return in_domain_flags

def evaluate_new_smiles_AD(train_df, new_df, similarity_threshold, count_threshold=2, similarity_metric='cosine'):

    num_train = len(train_df) 
    new_smiles = new_df['SMILES']
    new_struct_feat = np.array([generate_features(smiles) for smiles in new_smiles if generate_features(smiles).size > 0])
    new_struct_feat = np.nan_to_num(new_struct_feat)
    new_tabular = new_df.drop(['Name','SMILES'], axis=1)

    # 拼接结构+非结构特征

    new_all_feat = np.concatenate((new_struct_feat, new_tabular.values), axis=1)

    # 统一标准化
    scaler = StandardScaler()
    all_feat_combined = np.vstack((train_df, new_all_feat))
    all_feat_scaled = scaler.fit_transform(all_feat_combined)

    # 拆分回训练集和新集
    train_feat_scaled = all_feat_scaled[:num_train]
    new_feat_scaled = all_feat_scaled[num_train:]

    # 判断是否在应用域
    ad_flags = check_applicability_domain(train_feat_scaled, new_feat_scaled, similarity_metric, similarity_threshold, count_threshold)

    # 结果输出
    result_df = new_df.copy()
    result_df['in_AD_IE(-)'] = ad_flags
    return result_df

DEFAULT_SIMILARITY_THRESHOLD = 0.35
DEFAULT_COUNT_THRESHOLD = 1

def evaluate_ad(input_file, output_file, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, count_threshold=DEFAULT_COUNT_THRESHOLD):


    train_df = np.load("logIE_neg_train_features.npy")
    new_df = pd.read_excel(input_file)  # 新化合物数据

    result_df = evaluate_new_smiles_AD(train_df, new_df, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, count_threshold=DEFAULT_COUNT_THRESHOLD)
    result_df.to_excel(output_file, index=False)
    pass
