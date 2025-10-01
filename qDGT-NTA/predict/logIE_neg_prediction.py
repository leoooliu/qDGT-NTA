# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 18:49:37 2025

@author: 79234
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:30:08 2025

@author: 79234
"""

import joblib
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
import pandas as pd






# 初始化Mordred描述符计算器
calc = Calculator(descriptors, ignore_3D=True)

def generate_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.array([])  # 返回一个空数组处理无效分子的情况

    # RDKit 描述符
    
    rdkit_desc = np.array([desc[1](mol) for desc in Descriptors._descList])
    scaler = StandardScaler()
    rdkit_desc= scaler.fit_transform(rdkit_desc.reshape(-1, 1)).flatten()
    
    # 分子指纹
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_array1 = np.zeros((512,), dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint, fp_array1)
    
    MACCS =MACCSkeys.GenMACCSKeys(mol)           
    fp_arry2 = np.zeros((166))  
    DataStructs.ConvertToNumpyArray(MACCS,fp_arry2)
    
 

    # Hashed Morgan 指纹
    hashed_morgan_fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, radius=3, nBits=1024)
    hashed_morgan_array = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(hashed_morgan_fp, hashed_morgan_array)
    

    
    
    AFP = pyAvalonTools.GetAvalonFP(mol,512)
    fp_arry3 = np.zeros((1024,))
    DataStructs.ConvertToNumpyArray(AFP,fp_arry3)

    # Mordred 描述符
    mordred_desc = calc(mol)
    mordred_features = [x for x in mordred_desc.fill_missing().values()]

    # 融合特征
    features = np.concatenate([hashed_morgan_array, mordred_features])
    
   
  

    return features


def predict_smiles(input_file,output_file):    
    model = joblib.load('models/logIE_neg_XGBOOST.pkl')
    new_data = pd.read_excel(input_file)
    new_smiles_list = new_data['SMILES']
    features1 = np.array([generate_features(smiles) for smiles in new_smiles_list if generate_features(smiles).size > 0])
    features2 = new_data.drop(['Name', 'SMILES'], axis=1) 
    new_features_combined = np.concatenate((features1, features2), axis=1)
    predictions = model.predict(new_features_combined)
    # 保存预测结果
    new_data['Predicted_logIE_neg'] = predictions
    new_data.to_excel(output_file, index=False)
    pass


