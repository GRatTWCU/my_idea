import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import argparse
import os
import time
import pickle
import subprocess
import math
import logging
from typing import Tuple, Optional, Dict, Any

# 既存のSocial-LSTMモジュールをインポート
from model import TwoStageTrajectoryPredictor, TrajectoryPredictionTrainer
from utils import DataLoader
from helper import *
from grid import getSequenceGridMask

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class TrainingConfig:
    """訓練設定クラス"""
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # パス設定
        self.prefix = 'drive/semester_project/social_lstm_final/' if args.drive else ''
        self.f_prefix = 'drive/semester_project/social_lstm_final' if args.drive else '.'
        
        # モデル名とメソッド名
        self.method_name = "TwoStageModel"
        self.model_name = "IDEA1"
        
        # ディレクトリ設定
        self.log_directory = os.path.join(self.prefix, 'log', self.method_name, self.model_name)
        self.save_directory = os.path.join(self.prefix, 'model', self.method_name, self.model_name)
        
        # バリデーション設定
        self.validation_epoch_list = self._create_validation_schedule()
        
    def _create_validation_schedule(self) -> list:
        """バリデーションスケジュール作成"""
        freq = np.clip(self.args.freq_validation, 0, self.args.num_epochs)
        validation_epochs = list(range(freq, self.args.num_epochs + 1, freq))
        if validation_epochs:
            validation_epochs[-1] -= 1
        return validation_epochs
    
    def setup_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.log_directory, exist_ok=True)
        os.makedirs(self.save_directory, exist_ok=True)
        
        # ログディレクトリが存在しない場合のスクリプト実行
        if not os.path.isdir("log/"):
            logger.info("Creating directories...")
            try:
                subprocess.call([f'{self.f_prefix}/make_directories.sh'])
            except Exception as e:
                logger.warning(f"Failed to run directory creation script: {e}")

class MetricsCalculator:
    """評価指標計算クラス"""
    
    @staticmethod
    def calculate_ade(pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> float:
        """Average Displacement Error計算"""
        displacement_errors = torch.norm(pred - target, dim=-1)  # (batch, seq_len, num_peds)
        
        if valid_mask is not None:
            displacement_errors = displacement_errors * valid_mask
            return displacement_errors.sum() / valid_mask.sum()
        else:
            return displacement_errors.mean().item()
    
    @staticmethod
    def calculate_fde(pred: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> float:
        """Final Displacement Error計算"""
        final_displacement = torch.norm(pred[:, -1] - target[:, -1], dim=-1)  # (batch, num_peds)
        
        if valid_mask is not None:
            final_mask = valid_mask[:, -1]  # 最終フレームのマスク
            return (final_displacement * final_mask).sum() / final_mask.sum()
        else:
            return final_displacement.mean().item()

class DataProcessor:
    """データ処理クラス"""
    
    def __init__(self, max_pedestrians: int = 20):
        self.max_pedestrians = max_pedestrians
    
    def create_obstacle_map(self, dataset_dims: Tuple[int, int], 
                          current_positions: torch.Tensor) -> torch.Tensor:
        """環境情報マップを作成"""
        batch_size = current_positions.shape[0]
        width, height = dataset_dims
        
        # 位置を正規化
        x_norm = current_positions[:, 0] / width if width > 0 else 0.5
        y_norm = current_positions[:, 1] / height if height > 0 else 0.5
        
        # 正規化された座標を環境特徴として使用
        obstacle_map = torch.stack([x_norm, y_norm], dim=1)
        return obstacle_map.to(current_positions.device)
    
    def convert_to_batch_format(self, x_seq: torch.Tensor, 
                              lookup_seq: dict) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """ETH形式をバッチ形式に変換"""
        seq_len, actual_peds, coord_dim = x_seq.shape
        batch_size = 1
        
        # パディングして固定サイズにする
        padded_seq = torch.zeros(batch_size, seq_len, self.max_pedestrians, coord_dim, 
                               device=x_seq.device, dtype=x_seq.dtype)
        
        # マスクを作成（有効な歩行者位置を示す）
        valid_mask = torch.zeros(batch_size, seq_len, self.max_pedestrians, 
                               device=x_seq.device, dtype=torch.bool)
        
        # 実際の歩行者データをコピー
        num_peds_to_copy = min(actual_peds, self.max_pedestrians)
        if num_peds_to_copy > 0:
            padded_seq[0, :, :num_peds_to_copy, :] = x_seq[:, :num_peds_to_copy, :]
            valid_mask[0, :, :num_peds_to_copy] = True
        
        return padded_seq, num_peds_to_copy, valid_mask

class TwoStageTrainer:
    """2段階モデル訓練クラス"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.args = config.args
        self.data_processor = DataProcessor(self.args.max_pedestrians)
        self.metrics_calc = MetricsCalculator()
        
        # ログファイル設定
        self.setup_logging()
        
        # データローダー作成
        self.dataloader = self._create_dataloader()
        
        # モデル作成
        self.model = self._create_model()
        
        # オプティマイザー設定
        self.optimizer, self.scheduler = self._setup_optimizer()
        
        # 訓練状態
        self.best_val_loss = float('inf')
        self.best_ade = float('inf')
        self.best_epoch = 0
    
    def setup_logging(self):
        """ログファイル設定"""
        self.log_file_curve = open(
            os.path.join(self.config.log_directory, 'log_curve.txt'), 'w+'
        )
        self.log_file = open(
            os.path.join(self.config.log_directory, 'val.txt'), 'w+'
        )
    
    def _create_dataloader(self) -> DataLoader:
        """データローダー作成"""
        logger.info("Creating DataLoader...")
        return DataLoader(
            self.config.f_prefix, 
            self.args.batch_size, 
            self.args.seq_length, 
            self.args.num_validation, 
            forcePreProcess=True
        )
    
    def _create_model(self) -> nn.Module:
        """モデル作成"""
        logger.info("Creating Two-Stage Trajectory Prediction Model...")
        model = TwoStageTrajectoryPredictor(
            input_dim=2,
            hidden_dim=self.args.hidden_dim,
            output_dim=2,
            seq_len=self.args.seq_length,
            pred_len=self.args.pred_length,
            num_pedestrians=self.args.max_pedestrians
        )
        
        if self.args.use_cuda:
            model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        
        return model
    
    def _setup_optimizer(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """オプティマイザー設定"""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            weight_decay=self.args.lambda_param
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=self.args.freq_optimizer, 
            gamma=self.args.decay_rate
        )
        return optimizer, scheduler
    
    def save_config(self):
        """設定保存"""
        config_path = os.path.join(self.config.save_directory, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.args, f)
        logger.info(f"Configuration saved to {config_path}")
    
    def checkpoint_path(self, identifier: str) -> str:
        """チェックポイントパス生成"""
        return os.path.join(self.config.save_directory, f'two_stage_model_{identifier}.tar')
    
    def process_sequence(self, x_seq, y_seq, d_seq, numPedsList_seq, PedsList_seq, 
                        training: bool = True) -> Optional[Dict[str, float]]:
        """単一シーケンス処理"""
        try:
            # データセット次元取得
            folder_name = self.dataloader.get_directory_name_with_pointer(d_seq)
            dataset_data = self.dataloader.get_dataset_dimension(folder_name)
            
            # 密ベクトル変換
            x_dense, lookup_seq = self.dataloader.convert_proper_array(
                x_seq, numPedsList_seq, PedsList_seq
            )
            y_dense, _ = self.dataloader.convert_proper_array(
                y_seq, numPedsList_seq, PedsList_seq
            )
            
            if x_dense.shape[1] == 0:  # 歩行者がいない場合はスキップ
                return None
            
            # GPU転送
            x_dense = x_dense.to(device)
            y_dense = y_dense.to(device)
            
            # バッチ形式に変換
            x_batch, num_actual_peds, x_mask = self.data_processor.convert_to_batch_format(
                x_dense, lookup_seq
            )
            y_batch, _, y_mask = self.data_processor.convert_to_batch_format(
                y_dense, lookup_seq
            )
            
            # 環境マップ作成（最後の観測位置を使用）
            if x_dense.shape[1] > 0:
                last_positions = x_dense[-1, :, :]  # (num_peds, 2)
                centroid = last_positions.mean(dim=0).unsqueeze(0)  # (1, 2)
                obstacle_map = self.data_processor.create_obstacle_map(dataset_data, centroid)
            else:
                obstacle_map = None
            
            # 予測実行
            final_pred, stage1_pred, contrast_loss = self.model(
                x_batch, obstacle_map, training=training
            )
            
            # 損失計算（有効な歩行者のみ）
            final_loss = self._masked_mse_loss(final_pred, y_batch, y_mask)
            stage1_loss = self._masked_mse_loss(stage1_pred, y_batch, y_mask)
            
            total_loss = final_loss + 0.3 * stage1_loss + 0.1 * contrast_loss
            
            # 評価指標計算
            ade = self.metrics_calc.calculate_ade(final_pred, y_batch, y_mask)
            fde = self.metrics_calc.calculate_fde(final_pred, y_batch, y_mask)
            
            losses = {
                'total': total_loss.item(),
                'final': final_loss.item(),
                'stage1': stage1_loss.item(),
                'contrast': contrast_loss.item(),
                'ade': ade,
                'fde': fde
            }
            
            # 訓練時は勾配更新
            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            
            return losses
            
        except RuntimeError as e:
            logger.warning(f"Error processing sequence: {e}")
            return None
    
    def _masked_mse_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                        mask: torch.Tensor) -> torch.Tensor:
        """マスクを考慮したMSE損失"""
        mse = F.mse_loss(pred, target, reduction='none')  # (batch, seq, peds, 2)
        mse = mse.mean(dim=-1)  # (batch, seq, peds) - 座標次元で平均
        
        # マスクを適用
        masked_mse = mse * mask.float()
        
        # 有効な要素の平均を計算
        return masked_mse.sum() / mask.sum().clamp(min=1)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポック訓練"""
        logger.info(f'****************Training epoch {epoch} beginning******************')
        
        # バリデーションデータセット切り替え
        if (hasattr(self.dataloader, 'additional_validation') and 
            self.dataloader.additional_validation and 
            (epoch-1) in self.config.validation_epoch_list):
            self.dataloader.switch_to_dataset_type(True)
        
        self.dataloader.reset_batch_pointer(valid=False)
        self.model.train()
        
        epoch_losses = {'total': 0, 'final': 0, 'stage1': 0, 'contrast': 0, 'ade': 0, 'fde': 0}
        num_sequences = 0
        
        for batch in range(self.dataloader.num_batches):
            start_time = time.time()
            
            # バッチデータ取得
            x, y, d, numPedsList, PedsList, target_ids = self.dataloader.next_batch()
            
            batch_losses = {'total': 0, 'final': 0, 'stage1': 0, 'contrast': 0, 'ade': 0, 'fde': 0}
            valid_sequences = 0
            
            # 各シーケンスを処理
            for sequence in range(self.dataloader.batch_size):
                losses = self.process_sequence(
                    x[sequence], y[sequence], d[sequence],
                    numPedsList[sequence], PedsList[sequence],
                    training=True
                )
                
                if losses is not None:
                    for key in batch_losses:
                        batch_losses[key] += losses[key]
                    valid_sequences += 1
            
            # バッチ平均損失
            if valid_sequences > 0:
                for key in batch_losses:
                    epoch_losses[key] += batch_losses[key] / valid_sequences
                num_sequences += 1
            
            # ログ出力
            if batch % 10 == 0:
                avg_loss = batch_losses['total'] / max(valid_sequences, 1)
                elapsed = time.time() - start_time
                logger.info(f'Batch {batch}/{self.dataloader.num_batches}, '
                           f'Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s')
        
        # エポック平均損失
        if num_sequences > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_sequences
        
        logger.info(f'Epoch {epoch} Training - '
                   f'Total Loss: {epoch_losses["total"]:.4f}, '
                   f'ADE: {epoch_losses["ade"]:.4f}, '
                   f'FDE: {epoch_losses["fde"]:.4f}')
        
        # ログ記録
        self.log_file_curve.write(f"Training epoch: {epoch} loss: {epoch_losses['total']:.6f}\n")
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Optional[Dict[str, float]]:
        """バリデーション"""
        if not (hasattr(self.dataloader, 'valid_num_batches') and 
                self.dataloader.valid_num_batches > 0):
            return None
        
        logger.info('****************Validation epoch beginning******************')
        
        self.model.eval()
        val_losses = {'total': 0, 'ade': 0, 'fde': 0}
        val_sequences = 0
        
        self.dataloader.reset_batch_pointer(valid=True)
        
        with torch.no_grad():
            for batch in range(self.dataloader.valid_num_batches):
                x, y, d, numPedsList, PedsList, target_ids = self.dataloader.next_valid_batch()
                
                for sequence in range(self.dataloader.batch_size):
                    losses = self.process_sequence(
                        x[sequence], y[sequence], d[sequence],
                        numPedsList[sequence], PedsList[sequence],
                        training=False
                    )
                    
                    if losses is not None:
                        val_losses['total'] += losses['total']
                        val_losses['ade'] += losses['ade']
                        val_losses['fde'] += losses['fde']
                        val_sequences += 1
        
        if val_sequences > 0:
            for key in val_losses:
                val_losses[key] /= val_sequences
            
            logger.info(f'Epoch {epoch} Validation - '
                       f'Loss: {val_losses["total"]:.4f}, '
                       f'ADE: {val_losses["ade"]:.4f}, '
                       f'FDE: {val_losses["fde"]:.4f}')
            
            # ベストモデル保存
            if val_losses['ade'] < self.best_ade:
                self.best_ade = val_losses['ade']
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'ade': self.best_ade,
                    'args': self.args
                }, self.checkpoint_path('best'))
                logger.info(f'New best model saved! ADE: {self.best_ade:.4f}')
            
            # ログ記録
            self.log_file_curve.write(
                f"Validation epoch: {epoch} loss: {val_losses['total']:.6f} "
                f"ade: {val_losses['ade']:.6f} fde: {val_losses['fde']:.6f}\n"
            )
            
            return val_losses
        
        return None
    
    def train(self):
        """メイン訓練ループ"""
        logger.info("Starting training...")
        
        for epoch in range(self.args.num_epochs):
            # 訓練
            train_losses = self.train_epoch(epoch)
            
            # 学習率スケジューラ更新
            self.scheduler.step()
            
            # バリデーション
            val_losses = self.validate_epoch(epoch)
            
            # 定期保存
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'args': self.args
                }, self.checkpoint_path(epoch))
        
        logger.info(f'Training completed! Best ADE: {self.best_ade:.4f} at epoch {self.best_epoch}')
        
        # ログファイル終了
        self.log_file_curve.close()
        self.log_file.close()

def main():
    parser = argparse.ArgumentParser()
    
    # モデルパラメータ
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    parser.add_argument('--grid', action="store_true", default=True,
                        help='Whether store grids and use further epoch')
    
    # Idea1固有のパラメータ
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for two-stage model')
    parser.add_argument('--max_pedestrians', type=int, default=20,
                        help='Maximum number of pedestrians in a scene')
    
    args = parser.parse_args()
    
    # 設定作成
    config = TrainingConfig(args)
    config.setup_directories()
    
    # 訓練器作成・実行
    trainer = TwoStageTrainer(config)
    trainer.save_config()
    trainer.train()

if __name__ == '__main__':
    main()