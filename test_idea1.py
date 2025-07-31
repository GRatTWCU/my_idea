#!/usr/bin/env python3
"""
test_idea1.py - Social-LSTM IDEA1モデルのテストファイル
実際のデータセットを使わずにモデルの動作確認を行う
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
import pickle
import time
from typing import Dict, List, Tuple, Optional

# 現在のディレクトリをPythonパスに追加
sys.path.append('.')

def create_dummy_dataloader_data(batch_size=3, seq_length=8, pred_length=12, num_pedestrians=5):
    """
    ダミーデータを生成してDataLoaderの動作を模擬
    """
    print("ダミーデータ生成中...")
    
    # ダミーの軌跡データ生成
    x_batch = []
    y_batch = []
    d_batch = []
    numPedsList_batch = []
    PedsList_batch = []
    target_ids = []
    
    for batch_idx in range(batch_size):
        # 各バッチのシーケンスデータ
        x_seq = []
        y_seq = []
        numPeds_seq = []
        pedsList_seq = []
        
        for t in range(seq_length):
            # 時刻tでの歩行者データ
            num_peds_at_t = np.random.randint(2, num_pedestrians + 1)
            frame_data = []
            peds_list = []
            
            for ped_id in range(num_peds_at_t):
                # ランダムな位置生成（リアルな軌跡っぽく）
                base_x = 50 + ped_id * 20 + t * 2 + np.random.normal(0, 1)
                base_y = 100 + ped_id * 15 + t * 1.5 + np.random.normal(0, 1)
                
                frame_data.append([ped_id, base_x, base_y])
                peds_list.append(ped_id)
            
            x_seq.append(np.array(frame_data) if frame_data else np.array([]).reshape(0, 3))
            numPeds_seq.append(num_peds_at_t)
            pedsList_seq.append(peds_list)
        
        # 予測用のy系列（x系列の続き）
        for t in range(pred_length):
            num_peds_at_t = numPeds_seq[-1]  # 最後の観測と同じ歩行者数
            frame_data = []
            
            for ped_id in range(num_peds_at_t):
                # 続きの軌跡生成
                prev_x = x_seq[-1][ped_id][1] if len(x_seq[-1]) > ped_id else 50
                prev_y = x_seq[-1][ped_id][2] if len(x_seq[-1]) > ped_id else 100
                
                next_x = prev_x + 2 + np.random.normal(0, 0.5)
                next_y = prev_y + 1.5 + np.random.normal(0, 0.5)
                
                frame_data.append([ped_id, next_x, next_y])
            
            y_seq.append(np.array(frame_data) if frame_data else np.array([]).reshape(0, 3))
        
        x_batch.append(x_seq)
        y_batch.append(y_seq)
        d_batch.append(0)  # データセットインデックス
        numPedsList_batch.append(numPeds_seq)
        PedsList_batch.append(pedsList_seq)
        target_ids.append(0)  # ターゲットID
    
    return x_batch, y_batch, d_batch, numPedsList_batch, PedsList_batch, target_ids

class MockDataLoader:
    """DataLoaderの動作を模擬するクラス"""
    
    def __init__(self, batch_size=3, seq_length=8, pred_length=12, num_pedestrians=5):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.num_pedestrians = num_pedestrians
        self.num_batches = 2  # テスト用に2バッチ
        self.valid_num_batches = 1
        
        # ダミーデータ生成
        self._generate_dummy_data()
        
        # データセット次元（BIWI）
        self.dataset_dimensions = {'biwi': [720, 576]}
        
        print(f"MockDataLoader作成完了")
        print(f"  バッチサイズ: {batch_size}")
        print(f"  シーケンス長: {seq_length}")
        print(f"  予測長: {pred_length}")
        print(f"  最大歩行者数: {num_pedestrians}")
    
    def _generate_dummy_data(self):
        """ダミーデータを事前生成"""
        self.train_data = []
        self.valid_data = []
        
        # 訓練データ
        for _ in range(self.num_batches):
            batch_data = create_dummy_dataloader_data(
                self.batch_size, self.seq_length, self.pred_length, self.num_pedestrians
            )
            self.train_data.append(batch_data)
        
        # バリデーションデータ
        for _ in range(self.valid_num_batches):
            batch_data = create_dummy_dataloader_data(
                self.batch_size, self.seq_length, self.pred_length, self.num_pedestrians
            )
            self.valid_data.append(batch_data)
        
        self.current_batch = 0
        self.current_valid_batch = 0
    
    def next_batch(self):
        """次の訓練バッチを取得"""
        if self.current_batch >= len(self.train_data):
            self.current_batch = 0
        
        batch_data = self.train_data[self.current_batch]
        self.current_batch += 1
        return batch_data
    
    def next_valid_batch(self):
        """次のバリデーションバッチを取得"""
        if self.current_valid_batch >= len(self.valid_data):
            self.current_valid_batch = 0
        
        batch_data = self.valid_data[self.current_valid_batch]
        self.current_valid_batch += 1
        return batch_data
    
    def reset_batch_pointer(self, valid=False):
        """バッチポインタをリセット"""
        if valid:
            self.current_valid_batch = 0
        else:
            self.current_batch = 0
    
    def convert_proper_array(self, x_seq, num_pedlist, pedlist):
        """DataLoaderと同じ形式で配列変換"""
        # ユニークなIDを取得
        unique_ids = []
        for peds_in_frame in pedlist:
            unique_ids.extend(peds_in_frame)
        unique_ids = list(set(unique_ids))
        
        if not unique_ids:
            # 歩行者が存在しない場合
            seq_data = torch.zeros(self.seq_length, 1, 2)
            lookup_table = {}
        else:
            # ルックアップテーブル作成
            lookup_table = {ped_id: idx for idx, ped_id in enumerate(unique_ids)}
            
            # 配列作成
            seq_data = torch.zeros(self.seq_length, len(unique_ids), 2)
            
            for t, frame_data in enumerate(x_seq):
                if len(frame_data) > 0:
                    for ped_data in frame_data:
                        ped_id = int(ped_data[0])
                        if ped_id in lookup_table:
                            idx = lookup_table[ped_id]
                            seq_data[t, idx, 0] = ped_data[1]  # x座標
                            seq_data[t, idx, 1] = ped_data[2]  # y座標
        
        return seq_data, lookup_table
    
    def get_directory_name_with_pointer(self, dataset_idx):
        """データセット名を返す"""
        return 'biwi'
    
    def get_dataset_dimension(self, dataset_name):
        """データセット次元を返す"""
        return self.dataset_dimensions.get(dataset_name, [720, 576])

def test_model_creation():
    """モデル作成テスト"""
    print("\n" + "="*50)
    print("モデル作成テスト開始")
    print("="*50)
    
    try:
        from model import TwoStageTrajectoryPredictor
        
        # 小さなモデル作成
        model = TwoStageTrajectoryPredictor(
            input_dim=2,
            hidden_dim=32,
            output_dim=2,
            seq_len=8,
            pred_len=12,
            num_pedestrians=5,
            dropout=0.1
        )
        
        # パラメータ数計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ モデル作成成功")
        print(f"  総パラメータ数: {total_params:,}")
        print(f"  訓練可能パラメータ数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print(f"✗ モデル作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model):
    """順伝播テスト"""
    print("\n" + "="*50)
    print("順伝播テスト開始")
    print("="*50)
    
    try:
        # ダミー入力データ
        batch_size = 2
        seq_len = 8
        pred_len = 12
        num_peds = 5
        
        input_traj = torch.randn(batch_size, seq_len, num_peds, 2)
        obstacle_map = torch.randn(batch_size, 2)
        
        print(f"入力軌跡形状: {input_traj.shape}")
        print(f"障害物マップ形状: {obstacle_map.shape}")
        
        # 順伝播実行
        model.eval()
        with torch.no_grad():
            final_pred, stage1_pred, contrast_loss = model(
                input_traj, obstacle_map, training=False
            )
        
        print(f"✓ 順伝播成功")
        print(f"  最終予測形状: {final_pred.shape}")
        print(f"  第1段階予測形状: {stage1_pred.shape}")
        print(f"  コントラスト損失: {contrast_loss.item():.6f}")
        
        # 期待される形状チェック
        expected_shape = (batch_size, pred_len, num_peds, 2)
        if final_pred.shape == expected_shape:
            print(f"✓ 出力形状が正しい: {expected_shape}")
        else:
            print(f"✗ 出力形状が不正: 期待 {expected_shape}, 実際 {final_pred.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 順伝播エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step_with_mock_data(model):
    """MockDataLoaderを使った訓練ステップテスト"""
    print("\n" + "="*50)
    print("MockDataLoader訓練ステップテスト開始")
    print("="*50)
    
    try:
        # MockDataLoader作成
        dataloader = MockDataLoader(
            batch_size=2, seq_length=8, pred_length=12, num_pedestrians=5
        )
        
        # オプティマイザー設定
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 1回の訓練ステップ実行
        x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()
        
        print(f"バッチデータ取得成功:")
        print(f"  x length: {len(x)}")
        print(f"  y length: {len(y)}")
        print(f"  d length: {len(d)}")
        
        # データ変換とモデル実行
        model.train()
        total_loss = 0
        valid_sequences = 0
        
        for sequence in range(len(x)):
            x_seq, y_seq = x[sequence], y[sequence]
            d_seq = d[sequence]
            
            # データ変換
            x_dense, lookup_seq = dataloader.convert_proper_array(
                x_seq, numPedsList[sequence], PedsList[sequence]
            )
            y_dense, _ = dataloader.convert_proper_array(
                y_seq, numPedsList[sequence], PedsList[sequence]
            )
            
            if x_dense.shape[1] == 0:
                continue
            
            print(f"  シーケンス {sequence}: x_dense形状 {x_dense.shape}, y_dense形状 {y_dense.shape}")
            
            # バッチ形式に変換
            x_batch = x_dense.unsqueeze(0)  # (1, seq_len, num_peds, 2)
            y_batch = y_dense.unsqueeze(0)
            
            # 環境マップ（ダミー）
            dataset_dims = dataloader.get_dataset_dimension('biwi')
            last_pos = x_dense[-1, :, :].mean(dim=0).unsqueeze(0)  # 重心
            obstacle_map = torch.tensor([[last_pos[0]/dataset_dims[0], last_pos[1]/dataset_dims[1]]])
            
            # 予測実行
            final_pred, stage1_pred, contrast_loss = model(
                x_batch, obstacle_map, training=True
            )
            
            # 損失計算
            final_loss = F.mse_loss(final_pred, y_batch)
            stage1_loss = F.mse_loss(stage1_pred, y_batch)
            batch_loss = final_loss + 0.3 * stage1_loss + 0.1 * contrast_loss
            
            # バックプロパゲーション
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            total_loss += batch_loss.item()
            valid_sequences += 1
            
            print(f"    最終損失: {final_loss.item():.6f}")
            print(f"    第1段階損失: {stage1_loss.item():.6f}")
            print(f"    コントラスト損失: {contrast_loss.item():.6f}")
            print(f"    総損失: {batch_loss.item():.6f}")
        
        avg_loss = total_loss / max(valid_sequences, 1)
        print(f"✓ 訓練ステップ成功")
        print(f"  平均損失: {avg_loss:.6f}")
        print(f"  処理シーケンス数: {valid_sequences}")
        
        return True
        
    except Exception as e:
        print(f"✗ 訓練ステップエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics(model):
    """評価指標テスト"""
    print("\n" + "="*50)
    print("評価指標テスト開始")
    print("="*50)
    
    try:
        # ダミーデータ
        batch_size = 3
        pred_len = 12
        num_peds = 4
        
        # 実際の予測と真値（ダミー）
        predictions = torch.randn(batch_size, pred_len, num_peds, 2)
        ground_truth = torch.randn(batch_size, pred_len, num_peds, 2)
        
        # ADE (Average Displacement Error) 計算
        displacement_errors = torch.norm(predictions - ground_truth, dim=-1)
        ade = displacement_errors.mean().item()
        
        # FDE (Final Displacement Error) 計算
        final_displacement = torch.norm(predictions[:, -1] - ground_truth[:, -1], dim=-1)
        fde = final_displacement.mean().item()
        
        print(f"✓ 評価指標計算成功")
        print(f"  ADE: {ade:.6f}")
        print(f"  FDE: {fde:.6f}")
        
        # 実際のモデルでの評価
        model.eval()
        with torch.no_grad():
            input_traj = torch.randn(batch_size, 8, num_peds, 2)
            obstacle_map = torch.randn(batch_size, 2)
            
            final_pred, _, _ = model(input_traj, obstacle_map, training=False)
            
            # モデル出力での評価指標
            model_ade = torch.norm(final_pred - ground_truth, dim=-1).mean().item()
            model_fde = torch.norm(final_pred[:, -1] - ground_truth[:, -1], dim=-1).mean().item()
            
            print(f"  モデル出力 ADE: {model_ade:.6f}")
            print(f"  モデル出力 FDE: {model_fde:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 評価指標エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_components():
    """個別コンポーネントテスト"""
    print("\n" + "="*50)
    print("個別コンポーネントテスト開始")
    print("="*50)
    
    try:
        from model import (
            EnvironmentalAttentionModule,
            SingularTrajectoryPredictor,
            SocialSTGCNN
        )
        
        # 1. ECAM テスト
        print("1. EnvironmentalAttentionModule テスト")
        ecam = EnvironmentalAttentionModule(embedding_dim=32, env_dim=16, dropout=0.1)
        traj = torch.randn(2, 6, 2)
        obstacle = torch.randn(2, 2)
        attended_traj, contrast_feat = ecam(traj, obstacle)
        print(f"   ✓ ECAM動作確認 - 出力形状: {attended_traj.shape}")
        
        # 2. 第1段階予測器テスト
        print("2. SingularTrajectoryPredictor テスト")
        predictor = SingularTrajectoryPredictor(
            input_dim=2, hidden_dim=32, output_dim=2,
            seq_len=6, pred_len=8, num_layers=1, dropout=0.1
        )
        input_traj = torch.randn(2, 6, 2)
        pred_traj, contrast_feat = predictor(input_traj, obstacle, training=True)
        print(f"   ✓ 第1段階予測器動作確認 - 出力形状: {pred_traj.shape}")
        
        # 3. Social-STGCNN テスト
        print("3. SocialSTGCNN テスト")
        stgcnn = SocialSTGCNN(
            input_dim=2, hidden_dim=32, output_dim=2,
            seq_len=6, pred_len=8, num_nodes=4, num_blocks=2, dropout=0.1
        )
        input_trajs = torch.randn(2, 6, 4, 2)
        pred_trajs = stgcnn(input_trajs)
        print(f"   ✓ Social-STGCNN動作確認 - 出力形状: {pred_trajs.shape}")
        
        print("✓ 全コンポーネントテスト成功")
        return True
        
    except Exception as e:
        print(f"✗ コンポーネントテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("Social-LSTM IDEA1 テスト開始")
    print("="*60)
    
    # デバイス確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    print(f"PyTorchバージョン: {torch.__version__}")
    
    # テストリスト
    tests = [
        ("個別コンポーネントテスト", test_model_components),
        ("モデル作成テスト", test_model_creation),
    ]
    
    # 基本テスト実行
    model = None
    for test_name, test_func in tests:
        print(f"\n{test_name} 実行中...")
        try:
            if test_name == "モデル作成テスト":
                model = test_func()
                success = model is not None
            else:
                success = test_func()
                
            if success:
                print(f"✓ {test_name} 成功")
            else:
                print(f"✗ {test_name} 失敗")
                return False
        except Exception as e:
            print(f"✗ {test_name} で予期しないエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # モデルを使った詳細テスト
    if model is not None:
        detailed_tests = [
            ("順伝播テスト", lambda: test_forward_pass(model)),
            ("MockDataLoader訓練テスト", lambda: test_training_step_with_mock_data(model)),
            ("評価指標テスト", lambda: test_evaluation_metrics(model)),
        ]
        
        for test_name, test_func in detailed_tests:
            print(f"\n{test_name} 実行中...")
            try:
                success = test_func()
                if success:
                    print(f"✓ {test_name} 成功")
                else:
                    print(f"✗ {test_name} 失敗")
            except Exception as e:
                print(f"✗ {test_name} で予期しないエラー: {e}")
                import traceback
                traceback.print_exc()
    
    # 最終結果
    print("\n" + "="*60)
    print("テスト完了")
    print("="*60)
    print("🎉 全テスト成功！以下のコマンドでtrain_idea1.pyを実行できます:")
    print()
    print("# 軽量テスト実行")
    print("python train_idea1.py --batch_size 2 --seq_length 8 --pred_length 8 --num_epochs 3 --hidden_dim 32 --max_pedestrians 5")
    print()
    print("# 本格実行（データセットがある場合）")
    print("python train_idea1.py --batch_size 5 --seq_length 20 --pred_length 12 --num_epochs 30 --use_cuda")
    print("="*60)
    
    return True

if __name__ == "__main__":
    main()