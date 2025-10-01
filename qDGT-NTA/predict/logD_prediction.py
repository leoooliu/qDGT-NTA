# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:12:59 2025

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
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array1 = np.zeros((1024,), dtype=int)
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
   
  

  
def predict_smiles(input_file,output_file):    
    model = joblib.load('models/-logD_XGBOOST.pkl')
    new_data = pd.read_excel(input_file)
    new_smiles_list = new_data['SMILES']
    features = np.array([generate_features(smiles) for smiles in new_smiles_list if generate_features(smiles).size > 0])
    # 进行预测
    predictions = model.predict(features)
    # 保存预测结果
    new_data['Predicted_logD'] = predictions
    new_data.to_excel(output_file, index=False)
    pass

