import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any

# デバッグ用ヘルパー関数
def debug_tensor(tensor, name="tensor"):
    """テンソルの統計情報を出力するデバッグ関数"""
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tensor.shape}, mean={tensor.mean().item():.6f}, "
          f"std={tensor.std().item():.6f}, min={tensor.min().item():.6f}, "
          f"max={tensor.max().item():.6f}, has_nan={torch.isnan(tensor).any().item()}")

class EnvironmentalAttentionModule(nn.Module):
    """環境認識型注意機構 (Environmental Context Attention Module: ECAM) - 修正版"""
    
    def __init__(self, embedding_dim: int = 64, env_dim: int = 32, dropout: float = 0.1):
        super(EnvironmentalAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.env_dim = env_dim
        
        # 環境情報エンコーダ（重み初期化追加）
        self.env_encoder = nn.Sequential(
            nn.Linear(2, env_dim),
            nn.LayerNorm(env_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(env_dim, env_dim),
            nn.LayerNorm(env_dim),
            nn.ReLU()
        )
        
        # 軌跡エンコーダ（重み初期化追加）
        self.traj_encoder = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # マルチヘッド注意機構
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 環境-軌跡融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim + env_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # コントラスト学習用プロジェクタ
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重み初期化メソッド - 0出力の主要原因対策"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初期化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, trajectory: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: (batch_size, seq_len, 2) - 軌跡データ
            obstacle_map: (batch_size, 2) - 環境情報
        Returns:
            attended_trajectory: 注意機構適用後の軌跡
            contrast_feature: コントラスト学習用特徴量
        """
        batch_size, seq_len, _ = trajectory.shape
        
        # デバッグ出力
        debug_tensor(trajectory, "input_trajectory")
        
        # 軌跡エンコーディング
        traj_encoded = self.traj_encoder(trajectory)
        debug_tensor(traj_encoded, "traj_encoded")
        
        if obstacle_map is not None:
            debug_tensor(obstacle_map, "obstacle_map")
            
            # 環境エンコーディング
            env_encoded = self.env_encoder(obstacle_map)
            debug_tensor(env_encoded, "env_encoded")
            
            env_expanded = env_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 軌跡と環境情報を融合
            fused_features = torch.cat([traj_encoded, env_expanded], dim=-1)
            debug_tensor(fused_features, "fused_features")
            
            attended_traj = self.fusion_layer(fused_features)
            debug_tensor(attended_traj, "attended_traj_before_attention")
            
            # セルフアテンション適用
            attended_traj, _ = self.multihead_attention(
                attended_traj, attended_traj, attended_traj
            )
            debug_tensor(attended_traj, "attended_traj_after_attention")
        else:
            attended_traj = traj_encoded
        
        # コントラスト学習用特徴量（時系列平均）
        contrast_feature = self.projector(attended_traj.mean(dim=1))
        debug_tensor(contrast_feature, "contrast_feature")
        
        return attended_traj, contrast_feature

class SingularTrajectoryPredictor(nn.Module):
    """単体軌跡予測器（デバッグ版）"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 2,
                 seq_len: int = 8, pred_len: int = 12, num_layers: int = 2, dropout: float = 0.1):
        super(SingularTrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = num_layers
        
        # 改良されたLSTMエンコーダ（双方向 + 複数層）
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True
        )
        
        # エンコーダ出力次元調整
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # デコーダLSTM
        self.decoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # 出力層（残差接続付き）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 環境認識型注意機構
        self.ecam = EnvironmentalAttentionModule(hidden_dim, dropout=dropout)
        
        # Teacher forcing用の確率（訓練時）
        self.teacher_forcing_ratio = 0.5
        
        # 重み初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重み初期化メソッド"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # LSTM input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # LSTM hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:  # LSTM biases
                nn.init.zeros_(param.data)
                # forget gate biasを1に設定（勾配消失対策）
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)
        
        # 線形層の初期化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_traj: torch.Tensor, 
                obstacle_map: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_traj: (batch_size, seq_len, 2)
            obstacle_map: (batch_size, 2)
            training: 訓練モードフラグ
        Returns:
            predicted_traj: (batch_size, pred_len, 2)
            contrast_feature: コントラスト学習用特徴量
        """
        batch_size = input_traj.shape[0]
        
        print(f"\n=== SingularTrajectoryPredictor Forward Pass ===")
        debug_tensor(input_traj, "input_traj")
        
        # 入力データの正規化（0出力の原因対策）
        if input_traj.abs().max() > 100:
            print("Warning: Large input values detected, normalizing...")
            input_traj = input_traj / input_traj.abs().max() * 10
        
        # エンコーダで軌跡を符号化
        encoded_seq, (h_n, c_n) = self.encoder_lstm(input_traj)
        debug_tensor(encoded_seq, "encoded_seq")
        debug_tensor(h_n, "encoder_h_n")
        debug_tensor(c_n, "encoder_c_n")
        
        # 双方向LSTM出力を射影
        encoded_seq = self.encoder_projection(encoded_seq)
        debug_tensor(encoded_seq, "projected_encoded_seq")
        
        # ECAM適用
        attended_seq, contrast_feature = self.ecam(encoded_seq, obstacle_map)
        debug_tensor(attended_seq, "attended_seq")
        
        # デコーダの初期状態（最後の隠れ状態を使用）
        h_n = h_n[-self.num_layers:].contiguous()  # 前方向のみ使用
        c_n = c_n[-self.num_layers:].contiguous()
        
        debug_tensor(h_n, "decoder_init_h")
        debug_tensor(c_n, "decoder_init_c")
        
        # 予測軌跡生成
        predicted_traj = []
        decoder_input = input_traj[:, -1:, :]  # 最後の観測点
        decoder_hidden = (h_n, c_n)
        
        print(f"Starting prediction loop for {self.pred_len} steps...")
        
        for t in range(self.pred_len):
            # デコーダステップ
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            debug_tensor(decoder_output, f"decoder_output_step_{t}")
            
            # 出力予測
            pred_step = self.output_layer(decoder_output)
            debug_tensor(pred_step, f"pred_step_{t}")
            
            predicted_traj.append(pred_step)
            
            # 次のステップの入力
            decoder_input = pred_step
        
        predicted_traj = torch.cat(predicted_traj, dim=1)
        debug_tensor(predicted_traj, "final_predicted_traj")
        
        return predicted_traj, contrast_feature

# テスト用のシンプルなデバッグ関数
def test_model_components():
    """モデルコンポーネントの個別テスト"""
    print("=== Testing Model Components ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # テスト用データ生成
    batch_size = 4
    seq_len = 8
    pred_len = 12
    
    # 意味のあるテストデータ（単純な直線運動）
    input_traj = torch.zeros(batch_size, seq_len, 2, device=device)
    for b in range(batch_size):
        for t in range(seq_len):
            input_traj[b, t, 0] = t * 0.5 + b  # x座標
            input_traj[b, t, 1] = t * 0.3 + b * 0.5  # y座標
    
    obstacle_map = torch.randn(batch_size, 2, device=device) * 2
    
    print("Test data created:")
    debug_tensor(input_traj, "test_input_traj")
    debug_tensor(obstacle_map, "test_obstacle_map")
    
    # モデルテスト
    print("\n=== Testing SingularTrajectoryPredictor ===")
    model = SingularTrajectoryPredictor(
        input_dim=2, hidden_dim=64, output_dim=2,
        seq_len=seq_len, pred_len=pred_len
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            pred_traj, contrast_feat = model(input_traj, obstacle_map, training=False)
            print("Model test successful!")
            debug_tensor(pred_traj, "test_pred_traj")
            debug_tensor(contrast_feat, "test_contrast_feat")
            
            # 勾配チェック
            model.train()
            pred_traj, contrast_feat = model(input_traj, obstacle_map, training=True)
            loss = pred_traj.mean()  # ダミー損失
            loss.backward()
            
            # 勾配統計
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if grad_norm == 0:
                        print(f"WARNING: Zero gradient in {name}")
            print(f"Total gradient norm: {total_grad_norm}")
            
        except Exception as e:
            print(f"Model test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_model_components()
