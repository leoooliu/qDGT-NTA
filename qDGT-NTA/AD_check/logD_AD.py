# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:10:26 2025

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
    
    rdkit_desc = np.array([desc[1](mol) for desc in Descriptors._descList])
    if np.isinf(rdkit_desc).any():
        print("警告：描述符中存在无限值，正在替换为NaN")
        rdkit_desc[np.isinf(rdkit_desc)] = np.nan
    
    # 检查极大值
    max_abs_value = np.nanmax(np.abs(rdkit_desc))
    if max_abs_value > 1e10:  # 可根据需要调整阈值
        print(f"警告：发现极大值（最大绝对值：{max_abs_value}）")
        rdkit_desc = np.clip(rdkit_desc, -1e10, 1e10)  # 限制在合理范围内
    
    # 处理剩余的NaN值（来自无限值替换）
    if np.isnan(rdkit_desc).any():
        print("警告：发现NaN值，正在用均值替换")
        rdkit_desc = np.nan_to_num(rdkit_desc, nan=np.nanmean(rdkit_desc))
    scaler = StandardScaler()
    rdkit_desc= scaler.fit_transform(rdkit_desc.reshape(-1, 1)).flatten()
    
    # 分子指纹
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array1 = np.zeros((512,), dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint, fp_array1)
    
    MACCS =MACCSkeys.GenMACCSKeys(mol)           
    fp_arry2 = np.zeros((166))  
    DataStructs.ConvertToNumpyArray(MACCS,fp_arry2)
    
 

    # Hashed Morgan 指纹
    hashed_morgan_fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=3, nBits=1024)
    hashed_morgan_array = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(hashed_morgan_fp, hashed_morgan_array)
    

    
    
    AFP = pyAvalonTools.GetAvalonFP(mol,1024)
    fp_arry3 = np.zeros((1024,))
    DataStructs.ConvertToNumpyArray(AFP,fp_arry3)

    # Mordred 描述符
    mordred_desc = calc(mol)
    mordred_features = [x for x in mordred_desc.fill_missing().values()]

    # 融合特征
    features = np.concatenate([rdkit_desc, fp_array1, fp_arry2, hashed_morgan_array])
   
  

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

def check_applicability_domain(train_feat, new_feat, similarity_metric='cosine', similarity_threshold=0.15, count_threshold=1):
    if similarity_metric == 'cosine':
        similarities = cosine_similarity(train_feat, new_feat)
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
    num_similar_train_samples = np.sum(similarities >= similarity_threshold, axis=0)
    in_domain_flags = num_similar_train_samples >= count_threshold
    return in_domain_flags

def evaluate_new_smiles_AD(train_features, new_df, similarity_threshold, count_threshold=2, similarity_metric='cosine'):
    num_train = len(train_features)

    # 生成新分子特征
    new_struct_feat, valid_index = generate_all_features(new_df['SMILES'])

    # 统一标准化
    scaler = StandardScaler()
    all_feat_combined = np.vstack((train_features, new_struct_feat))
    all_feat_scaled = scaler.fit_transform(all_feat_combined)

    # 拆分回训练集和新集
    train_feat_scaled = all_feat_scaled[:num_train]
    new_feat_scaled = all_feat_scaled[num_train:]

    # 判断是否在应用域
    ad_flags = check_applicability_domain(train_feat_scaled, new_feat_scaled,
                                          similarity_metric, similarity_threshold, count_threshold)

    # 输出结果（只保留有效分子）
    result_df = new_df.iloc[valid_index].copy()
    result_df['in_AD_D'] = ad_flags
    return result_df
DEFAULT_SIMILARITY_THRESHOLD = 0.15
DEFAULT_COUNT_THRESHOLD = 1

def evaluate_ad(input_file, output_file, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, count_threshold=DEFAULT_COUNT_THRESHOLD):


    train_df = np.load("-logD_train_features.npy")
    new_df = pd.read_excel(input_file)  # 新化合物数据

    result_df = evaluate_new_smiles_AD(train_df, new_df,similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, count_threshold=DEFAULT_COUNT_THRESHOLD)
    result_df.to_excel(output_file, index=False)
    pass

