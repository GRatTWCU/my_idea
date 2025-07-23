import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# --------------------- 環境特徴量抽出 ---------------------
def extract_environment_features(lidar_points, trajectories):
    """
    環境特徴量を抽出するダミー関数。
    実際にはLiDARデータや他の環境センサーデータから意味のある特徴量を抽出します。
    ここでは、モデルの入力要件を満たすためのプレースホルダーとして機能します。
    
    Args:
        lidar_points (torch.Tensor or None): ダミーまたは実際のLiDARポイントデータ。
                                            形状は (B, num_points, D) を想定。
        trajectories (torch.Tensor): 歩行者の軌跡データ。形状は (B, N_ped, T, D) を想定。

    Returns:
        tuple: static_density, nearest_dist, pedestrian_density, interaction_strength
               (全て torch.Tensor)
    """
    B, N_ped, T, _ = trajectories.shape
    
    # LiDARポイントがNoneの場合（ダミーデータセットの場合など）に対応
    lidar_points_count = lidar_points.size(1) if lidar_points is not None else 1000 

    # ダミーの環境特徴量を生成
    # 実際のアプリケーションでは、ここにLiDARデータなどから抽出した真の環境特徴量を実装します。
    static_density = torch.full((B, N_ped, 1), fill_value=float(lidar_points_count)/1000.0)
    nearest_dist = torch.rand(B, N_ped, 1) * 10
    pedestrian_density = torch.full((B, N_ped, 1), fill_value=float(N_ped)/10.0)
    interaction_strength = torch.rand(B, N_ped, 1)
    
    return static_density, nearest_dist, pedestrian_density, interaction_strength

# --------------------- GATモジュール ---------------------
class SimpleGATLayer(nn.Module):
    """
    シンプルなGraph Attention Network (GAT) レイヤー。
    歩行者間の相互作用をモデリングするために使用されます。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(out_features*2, 1, bias=False)

    def forward(self, h):
        """
        Args:
            h (torch.Tensor): 各ノード（歩行者）の特徴量。形状は (B, N, F_in)。
                              B: バッチサイズ, N: 歩行者数, F_in: 入力特徴量次元。
        Returns:
            torch.Tensor: Attentionを適用した後の各ノードの特徴量。形状は (B, N, F_out)。
        """
        B, N, F_in = h.size()
        Wh = self.fc(h) # 線形変換 (B, N, F_out)

        # Attentionスコア計算のための特徴量ペアを作成
        # Wh_repeat_i: 各ノードiの特徴量を他の全てのノードjに対して複製 (B, N, N, F_out)
        Wh_repeat_i = Wh.unsqueeze(2).repeat(1,1,N,1)
        # Wh_repeat_j: 各ノードjの特徴量を他の全てのノードiに対して複製 (B, N, N, F_out)
        Wh_repeat_j = Wh.unsqueeze(1).repeat(1,N,1,1)
        
        # 結合された特徴量からAttentionスコアを計算
        e = self.attn_fc(torch.cat([Wh_repeat_i, Wh_repeat_j], dim=-1)).squeeze(-1) # (B, N, N)
        
        # LeakyReLUを適用し、softmaxでAttention重みを正規化
        alpha = F.softmax(F.leaky_relu(e), dim=-1) # (B, N, N)
        
        # Attention重みを使って特徴量を集約
        h_prime = torch.bmm(alpha, Wh) # (B, N, F_out)
        return h_prime

# --------------------- Beam Model ---------------------
class BeamModel(nn.Module):
    """
    障害物回避パターンを学習するためのシンプルなBeam Model。
    環境特徴量から予測変位を生成します。
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 2D変位 (dx, dy) を出力
        )

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): 環境特徴量。形状は (B, N, input_dim)。
        Returns:
            torch.Tensor: 予測された2D変位。形状は (B, N, 2)。
        """
        return self.fc(features)

# --------------------- 環境適応型LSTM ---------------------
class EnvAdaptiveLSTM(nn.Module):
    """
    環境に適応した補正ベクトルを予測するためのLSTMモデル。
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 入力シーケンス（軌跡と環境特徴量の結合）。
                              形状は (Batch_size * Num_pedestrians, Sequence_length, Input_dim)。
        Returns:
            torch.Tensor: 予測された補正ベクトル。形状は (Batch_size * Num_pedestrians, Output_dim)。
        """
        # LSTMにシーケンスを入力
        # _: 出力シーケンス (ここでは使用しない)
        # hn: 最終隠れ状態 (num_layers * num_directions, batch_size, hidden_size)
        # cn: 最終セル状態 (num_layers * num_directions, batch_size, hidden_size)
        _, (hn, _) = self.lstm(x)
        
        # 最終レイヤーの隠れ状態を線形層に通して出力
        out = self.fc(hn[-1]) # hn[-1] は最後のレイヤーの隠れ状態
        return out

# --------------------- 統合モデル ---------------------
class TrajectoryPredictor(nn.Module):
    """
    歩行者軌跡予測のための統合モデル。
    GAT、Beam Model、環境適応型LSTMを組み合わせて予測を行います。
    """
    def __init__(self, gat_in_dim, gat_out_dim, beam_in_dim, beam_hidden_dim, lstm_input_dim, lstm_hidden_dim):
        super().__init__()
        self.gat = SimpleGATLayer(gat_in_dim, gat_out_dim)
        self.beam = BeamModel(beam_in_dim, beam_hidden_dim)
        self.env_lstm = EnvAdaptiveLSTM(lstm_input_dim, lstm_hidden_dim)

    def forward(self, traj_hist, env_feats):
        """
        Args:
            traj_hist (torch.Tensor): 過去の軌跡履歴。形状は (B, N, T_hist, 2)。
            env_feats (torch.Tensor): 環境特徴量。形状は (B, N, F_env)。
        Returns:
            tuple:
                - final_pred (torch.Tensor): 最終的な予測変位。形状は (B, N, 2)。
                - pred_confidence (torch.Tensor): 予測信頼度。形状は (B, N)。
        """
        B, N, T_hist, _ = traj_hist.shape

        # --- 第一段階予測 ---
        # GAT: 歩行者間相互作用モデリング
        # GATの入力は各歩行者の最終位置 (B, N, 2)
        gat_input = traj_hist[:, :, -1, :] # 最終時刻の(x,y)座標
        gat_out = self.gat(gat_input) # GATの出力 (B, N, gat_out_dim)

        # Beam Model: 障害物回避パターン学習
        # Beam Modelの入力は環境特徴量 (B, N, F_env)
        beam_out = self.beam(env_feats) # Beam Modelの出力 (B, N, 2) - 予測変位

        # 統合: 環境認識軌跡予測 (GAT出力の一部とBeam Modelの出力の合計)
        # gat_outの最初の2次元を予測変位として使用 (gat_out_dim >= 2 を想定)
        first_pred = gat_out[:, :, :2] + beam_out # 最初の予測変位 (B, N, 2)

        # --- 環境状況評価 ---
        # 予測信頼度計算 (Beam Modelの出力のノルムの負のシグモイド)
        pred_confidence = torch.sigmoid(-torch.norm(beam_out, dim=-1, keepdim=True)) # (B, N, 1)

        # 補正必要性判定 (信頼度が低い場合に補正が必要と判断)
        correction_needed = (pred_confidence < 0.5).float() # (B, N, 1) - 0 or 1

        # --- 第二段階補正 ---
        # 環境適応型LSTMの入力準備
        # LSTM入力は (B*N, T_hist, input_dim)
        
        # 環境特徴量を時系列に沿って繰り返す
        # env_feats: (B, N, F_env) -> unsqueeze(2) -> (B, N, 1, F_env) -> repeat(1,1,T_hist,1) -> (B, N, T_hist, F_env)
        lstm_in_env = env_feats.unsqueeze(2).repeat(1,1,T_hist,1).reshape(B*N, T_hist, -1)
        
        # 軌跡履歴と環境特徴量を結合
        # traj_hist: (B, N, T_hist, 2) -> reshape -> (B*N, T_hist, 2)
        lstm_in = torch.cat([traj_hist.reshape(B*N, T_hist, 2), lstm_in_env], dim=-1) # (B*N, T_hist, 2 + F_env)
        
        # 補正ベクトル予測
        correction = self.env_lstm(lstm_in).view(B, N, 2) # LSTMの出力 (B, N, 2)

        # 最終軌跡出力 (最初の予測に変位補正を適用)
        # correction_needed は (B, N, 1) なので、correction (B, N, 2) とブロードキャストで乗算
        final_pred = first_pred + correction * correction_needed # 最終予測変位 (B, N, 2)
        
        return final_pred, pred_confidence.squeeze(-1) # pred_confidenceは (B, N) になる

# --------------------- ADE/FDE計算 ---------------------
def calculate_ade_fde(pred, gt):
    """
    Average Displacement Error (ADE) と Final Displacement Error (FDE) を計算します。
    このモデルは単ステップ予測のため、ADEとFDEは同じ値になります。
    
    Args:
        pred (torch.Tensor): 予測された軌跡または位置。形状は (B, N, 2)。
        gt (torch.Tensor): 正解の軌跡または位置。形状は (B, N, 2)。
    Returns:
        tuple: ADE (float), FDE (float)
    """
    diff = pred - gt # 予測と正解の差 (B, N, 2)
    dist = torch.norm(diff, dim=-1) # 各予測のユークリッド距離 (B, N)
    ade = dist.mean().item() # 全ての予測と正解間の平均距離
    fde = ade  # 単ステップ予測の場合、ADEとFDEは同じ
    return ade, fde

# --------------------- ETH形式データセット ---------------------
class ETHFormattedDataset(Dataset):
    """
    complete_nuscenes_setup.py によって生成されたETH形式のデータセット (.txt ファイル) を読み込みます。
    各行は 'frame_id person_id x y' の形式です。
    """
    def __init__(self, data_dir, obs_len=4, pred_len=1): # pred_lenをモデルの出力に合わせて1に設定
        """
        Args:
            data_dir (str): train/val/test などのディレクトリパス。
            obs_len (int): 観測する過去のステップ数。
            pred_len (int): 予測する未来のステップ数。
        """
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.sequences = [] # 各要素は (history_traj, future_traj_gt, dummy_env_feats)

        # data_dir内の全ての.txtファイルを探索
        scene_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
        
        if not scene_files:
            print(f"Warning: No .txt files found in {data_dir}. Please ensure data conversion was successful.")
            print(f"Expected data in: {os.path.abspath(data_dir)}")

        for scene_file in scene_files:
            # 各シーンファイルからデータを読み込み
            # ETH形式: frame_id person_id x y
            try:
                data = np.loadtxt(scene_file, dtype=np.float32)
            except Exception as e:
                print(f"Error loading {scene_file}: {e}")
                continue # このファイルはスキップ

            if data.ndim == 1: # 単一の行しかない場合、2次元配列に変換
                data = np.expand_dims(data, axis=0)
            if data.shape[0] == 0: # データがない場合
                continue

            # frame_idでソート
            data = data[data[:, 0].argsort()]

            # 各人物の軌跡を抽出
            ped_ids = np.unique(data[:, 1]).astype(int)
            
            for ped_id in ped_ids:
                ped_data = data[data[:, 1] == ped_id]
                
                # 軌跡が十分な長さを持つか確認 (obs_len + pred_len)
                if len(ped_data) >= (self.obs_len + self.pred_len):
                    # スライディングウィンドウでシーケンスを生成
                    for i in range(len(ped_data) - (self.obs_len + self.pred_len) + 1):
                        sequence = ped_data[i : i + self.obs_len + self.pred_len, 2:4] # x,y座標のみ
                        
                        history_traj = torch.tensor(sequence[:self.obs_len], dtype=torch.float32)
                        future_traj_gt = torch.tensor(sequence[self.obs_len:], dtype=torch.float32)

                        # 環境特徴量はダミーを生成
                        # 実際には、LiDARや他の環境データから抽出する必要があります
                        # ここでは、単一の歩行者に対するダミー特徴量として (1, F_env) の形状で生成
                        # TrajectoryPredictorは (B, N, F_env) を期待するため、N=1として扱う
                        dummy_lidar_points = torch.randn(1, 1000, 3) # ダミーLiDAR (B=1, num_points, 3)
                        dummy_trajectories_for_env = history_traj.unsqueeze(0).unsqueeze(0) # (B=1, N=1, T_hist, 2)

                        static_density, nearest_dist, pedestrian_density, interaction_strength = \
                            extract_environment_features(dummy_lidar_points, dummy_trajectories_for_env)
                        
                        # env_feats は (B=1, N=1, F_env) なので、squeezeして (F_env) にする
                        # ETHFormattedDatasetは各シーケンスに対して単一の環境特徴量ベクトルを返す
                        dummy_env_feats = torch.cat([static_density.squeeze(0).squeeze(0), 
                                                     nearest_dist.squeeze(0).squeeze(0), 
                                                     pedestrian_density.squeeze(0).squeeze(0), 
                                                     interaction_strength.squeeze(0).squeeze(0)], dim=-1)

                        self.sequences.append((history_traj, future_traj_gt, dummy_env_feats))
                        
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# --------------------- 訓練関数 ---------------------
def train_model(model, dataloader, optimizer, criterion, num_epochs, device):
    """
    モデルを訓練する関数。
    """
    model.train() # モデルを訓練モードに設定
    print("--- 訓練開始 ---")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (history_traj, future_traj_gt, env_feats) in enumerate(dataloader):
            # データをデバイスに転送 (GPUがあればGPUへ)
            history_traj = history_traj.to(device) # (B, T_hist, 2)
            future_traj_gt = future_traj_gt.to(device) # (B, T_pred, 2)
            env_feats = env_feats.to(device) # (B, F_env)

            # モデルの入力形式に合わせる (TrajectoryPredictorは (B, N, T, D) を期待)
            # ETHFormattedDatasetは各シーケンスで1人の歩行者を扱うため、N=1としてunsqueeze(1)
            history_traj_input = history_traj.unsqueeze(1) # (B, 1, T_hist, 2)
            env_feats_input = env_feats.unsqueeze(1) # (B, 1, F_env)

            optimizer.zero_grad() # 勾配をゼロにリセット

            # 順伝播
            # TrajectoryPredictorは単ステップ予測 (B, N, 2) を出力
            pred_traj_single_step, _ = model(history_traj_input, env_feats_input) 

            # 損失計算 (ここではMSEを使用)
            # future_traj_gt は (B, T_pred, 2) なので、最初の予測ステップ (B, 2) と比較
            # pred_traj_single_step は (B, 1, 2) なので、squeeze(1) して (B, 2) にする
            loss = criterion(pred_traj_single_step.squeeze(1), future_traj_gt[:, 0, :])
            
            # 逆伝播
            loss.backward()
            optimizer.step() # パラメータ更新

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0: # 10バッチごとにログ出力
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] 完了, 平均損失: {avg_loss:.4f}")
    print("--- 訓練完了 ---")

# --------------------- 評価関数 ---------------------
def evaluate_model(model, dataloader, device):
    """
    モデルを評価する関数。
    """
    model.eval() # モデルを評価モードに設定
    total_ade = 0
    total_fde = 0
    num_samples = 0
    print("--- 評価開始 ---")
    with torch.no_grad(): # 勾配計算を無効化
        for batch_idx, (history_traj, future_traj_gt, env_feats) in enumerate(dataloader):
            history_traj = history_traj.to(device)
            future_traj_gt = future_traj_gt.to(device)
            env_feats = env_feats.to(device)

            history_traj_input = history_traj.unsqueeze(1)
            env_feats_input = env_feats.unsqueeze(1)
            
            pred_traj, _ = model(history_traj_input, env_feats_input)
            
            # ADE/FDE計算 (単ステップ予測なので、gt_next_posはfuture_traj_gtの最初のステップ)
            gt_next_pos = future_traj_gt[:, 0, :].unsqueeze(1) # (B, 1, 2)
            
            # calculate_ade_fde は (B, N, D) を期待するので、pred_trajとgt_next_posは (B, 1, 2)
            ade, fde = calculate_ade_fde(pred_traj, gt_next_pos) 
            
            total_ade += ade * history_traj.size(0) # バッチサイズで重み付け
            total_fde += fde * history_traj.size(0)
            num_samples += history_traj.size(0)

    avg_ade = total_ade / num_samples
    avg_fde = total_fde / num_samples
    print(f"--- 評価完了 ---")
    print(f"平均ADE (1-step): {avg_ade:.4f}, 平均FDE (1-step): {avg_fde:.4f}")
    return avg_ade, avg_fde


# --------------------- メイン ---------------------
def main():
    # complete_nuscenes_setup.py で変換されたデータを使用します。
    # まず、complete_nuscenes_setup.py を --mode raw (または --mode dummy) で実行して、
    # './datasets/nuscenes_mini/train/', './datasets/nuscenes_mini/val/', './datasets/nuscenes_mini/test/'
    # ディレクトリにデータが生成されていることを確認してください。
    
    # データセットのパス
    train_data_path = "./datasets/nuscenes_mini/train/"
    val_data_path = "./datasets/nuscenes_mini/val/" # 検証用データパスも追加
    
    # 観測ステップ数 (過去の軌跡の長さ) と予測ステップ数 (未来の軌跡の長さ) を定義
    # このモデルは単ステップ予測のため、pred_lenは1に設定します。
    obs_len = 4 
    pred_len = 1 # モデルの出力が単一の次の位置なので、予測長は1

    # 訓練データセットとデータローダーの準備
    train_dataset = ETHFormattedDataset(train_data_path, obs_len=obs_len, pred_len=pred_len)
    val_dataset = ETHFormattedDataset(val_data_path, obs_len=obs_len, pred_len=pred_len)
    
    if len(train_dataset) == 0:
        print(f"エラー: 訓練データセットが空です。'{train_data_path}' にデータファイルが存在しないか、")
        print("complete_nuscenes_setup.py の実行が成功していない可能性があります。")
        return
    if len(val_dataset) == 0:
        print(f"警告: 検証データセットが空です。'{val_data_path}' にデータファイルが存在しないか、")
        print("complete_nuscenes_setup.py の実行で検証データが生成されていない可能性があります。")
        # 検証データがない場合は、訓練データで代用するか、評価をスキップするなどの対応が必要
        # ここでは評価をスキップします
        run_validation = False
    else:
        run_validation = True


    print(f"訓練データセット内のシーケンス数: {len(train_dataset)}")
    if run_validation:
        print(f"検証データセット内のシーケンス数: {len(val_dataset)}")
    
    batch_size = 32 # バッチサイズを設定
    # num_workers > 0 はデータロードを並列化しますが、Windowsでは問題が発生することがあります。
    # まずは num_workers=0 で試すことを推奨します。
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # 検証データはシャッフルしない

    # モデルの初期化
    # GATの入力は軌跡の最後の2次元 (x,y) なので gat_in_dim=2
    # 環境特徴量は extract_environment_features の出力の結合なので dimension = 1+1+1+1 = 4
    # LSTMの入力は軌跡 (2) と環境特徴量 (4) の結合なので 2+4 = 6
    model = TrajectoryPredictor(
        gat_in_dim=2, gat_out_dim=4, # gat_out_dim は GATの出力特徴量次元
        beam_in_dim=4, beam_hidden_dim=16, # beam_in_dim は環境特徴量の次元
        lstm_input_dim=2+4, lstm_hidden_dim=32 # lstm_input_dim は軌跡次元 + 環境特徴量次元
    )

    # デバイス設定 (GPUがあればGPUを使用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用デバイス: {device}")

    # 最適化手法と損失関数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # 損失関数 (ここではMSEを使用)

    num_epochs = 10 # エポック数

    # モデルの訓練
    train_model(model, train_dataloader, optimizer, criterion, num_epochs, device)

    # モデルの評価
    if run_validation:
        print("\n--- 検証データでの評価 ---")
        evaluate_model(model, val_dataloader, device)
    else:
        print("\n検証データセットが空のため、評価をスキップします。")


if __name__ == "__main__":
    main()
