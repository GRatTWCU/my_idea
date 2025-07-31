import torch
import numpy as np

def getSequenceGridMask(frame_data, dimensions, neighborhood_size, grid_size):
    """
    グリッドマスクを生成（簡易版）
    
    Args:
        frame_data: フレームデータ
        dimensions: データセット次元 [width, height]
        neighborhood_size: 近傍サイズ
        grid_size: グリッドサイズ
    
    Returns:
        grid_mask: グリッドマスク
    """
    if not isinstance(frame_data, (list, tuple)):
        frame_data = [frame_data]
    
    batch_size = len(frame_data)
    
    # 簡易実装：実際のSocial-LSTMのgrid.pyを使用することを推奨
    grid_masks = []
    
    for frame in frame_data:
        # 空のグリッドマスクを作成
        grid_mask = torch.zeros(grid_size, grid_size)
        
        # フレームデータが存在する場合の処理
        if hasattr(frame, 'shape') and len(frame.shape) > 0:
            # フレーム内の歩行者位置に基づいてグリッドを更新
            if len(frame.shape) >= 2 and frame.shape[0] > 0:
                positions = frame[:, 1:3] if frame.shape[1] >= 3 else frame
                
                # 位置を正規化してグリッドインデックスに変換
                if len(dimensions) >= 2:
                    normalized_x = positions[:, 0] / dimensions[0] * (grid_size - 1)
                    normalized_y = positions[:, 1] / dimensions[1] * (grid_size - 1)
                    
                    # グリッド範囲内のインデックスのみ処理
                    valid_indices = (
                        (normalized_x >= 0) & (normalized_x < grid_size) &
                        (normalized_y >= 0) & (normalized_y < grid_size)
                    )
                    
                    if torch.any(valid_indices):
                        grid_x = normalized_x[valid_indices].long()
                        grid_y = normalized_y[valid_indices].long()
                        grid_mask[grid_y, grid_x] = 1.0
        
        grid_masks.append(grid_mask)
    
    return torch.stack(grid_masks) if len(grid_masks) > 1 else grid_masks[0]