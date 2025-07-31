import os
import numpy as np
import torch

def unique_list(input_list):
    """リストから重複を除去"""
    return list(set(input_list))

def get_all_file_names(directory):
    """ディレクトリ内の全ファイル名を取得"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) 
            if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]

def create_directory_if_not_exists(directory):
    """ディレクトリが存在しない場合は作成"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize_coordinates(coords, dataset_dimensions):
    """座標の正規化"""
    if len(dataset_dimensions) >= 2:
        coords[:, 0] /= dataset_dimensions[0]  # x座標
        coords[:, 1] /= dataset_dimensions[1]  # y座標
    return coords