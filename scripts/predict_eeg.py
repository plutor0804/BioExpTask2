import os
import time
import torch
import numpy as np
import joblib
import pyautogui
from pynput.keyboard import Controller, Key

# === 遊戲與終端機位置座標（需手動量測） ===
CLICK_MAZE_X = 357
CLICK_MAZE_Y = 511
CLICK_TERMINAL_X = 1734
CLICK_TERMINAL_Y = 1005

def focus_maze_window():
    try:
        pyautogui.click(CLICK_MAZE_X, CLICK_MAZE_Y)
        print("切換至迷宮視窗")
    except Exception as e:
        print(f"❌ 切換失敗：{e}")

def focus_terminal_window():
    try:
        pyautogui.click(CLICK_TERMINAL_X, CLICK_TERMINAL_Y)
        print("切換至終端機視窗")
    except Exception as e:
        print(f"❌ 切換失敗：{e}")

# === CNN + 小型 MLP 模型架構 ===
class HybridCNN(torch.nn.Module):
    def __init__(self, vec_dim, meta_dim, num_classes):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(meta_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 + 64, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        vec = x[:, :496].unsqueeze(1)
        meta = x[:, 496:]
        cnn_out = self.cnn(vec)
        mlp_out = self.mlp(meta)
        return self.classifier(torch.cat([cnn_out, mlp_out], dim=1))

# === 特徵處理函數 ===
def compute_band_ratios(band_feat):
    theta, alpha, beta = band_feat[1], band_feat[2], band_feat[3]
    return np.array([
        theta / (alpha + 1e-8),
        alpha / (beta + 1e-8),
        beta / (theta + 1e-8)
    ])

def extract_band_power(vec, freqs):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50)
    }
    powers = []
    for (low, high) in bands.values():
        idx = np.where((freqs >= low) & (freqs < high))
        powers.append(np.mean(vec[idx]))
    return np.array(powers)


def extract_features(vec):
    vec_safe = vec + 1e-8
    return np.array([
        np.mean(vec),
        np.std(vec),
        np.sqrt(np.mean(vec ** 2)),
        np.min(vec),
        np.max(vec),
        np.percentile(vec, 25),
        np.median(vec),
        np.percentile(vec, 75),
        np.mean((vec - np.mean(vec))**3) / (np.std(vec)**3 + 1e-8),
        np.mean((vec - np.mean(vec))**4) / (np.std(vec)**4 + 1e-8),
        np.sum(vec**2),
        -np.sum((vec_safe/np.sum(vec_safe)) * np.log(vec_safe/np.sum(vec_safe)))
    ])

def load_single_txt_with_features(txt_path):
    fs = 500
    raw = np.loadtxt(txt_path)
    N = len(raw)
    if N < 10_000:
        raise ValueError(f"檔案只有 {N} 點，少於 10 000")

    start = (N - 5000) // 2      # 亦可寫成 N//2 - 5000
    raw = raw[start : start + 5000]   # 取正中間 10 秒資料 (500 Hz)


    # === FFT → log(1+power) ===
    n_fft      = (len(raw) - 1) * 2              # 19 998
    fft_result = np.fft.rfft(raw, n=n_fft)
    power      = np.abs(fft_result) ** 2
    log_power  = np.log1p(power)

    # 0.5–50 Hz 範圍
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    idx   = (freqs >= 0.5) & (freqs <= 50)
    log_power = log_power[idx]
    freqs     = freqs[idx]

    # **保留前 496 維**──與訓練一致
    if len(log_power) < 496:
        raise ValueError(f"頻域資料只有 {len(log_power)} 點，少於 496")
    log_power = log_power[:496]
    freqs     = freqs[:496]

    # 20 維輔助特徵
    stat_feat  = extract_features(log_power)
    band_feat  = extract_band_power(log_power, freqs)
    ratio_feat = compute_band_ratios(band_feat)

    # 496 + 20 = 516 維
    return np.concatenate([log_power, stat_feat, band_feat, ratio_feat])

# === 載入模型與 Scaler ===
VEC_DIM = 496           # log-power 部分
META_DIM = 20           # 統計+band+ratio
NUM_CLASSES = 4

model = HybridCNN(vec_dim=VEC_DIM, meta_dim=META_DIM, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("../models/model6/best_hybrid_model.pt", map_location=torch.device('cpu')))
model.eval()
scaler = joblib.load("../models/model6/best_scaler.pkl")

label_map = ["relax", "concentrating", "stress", "memory"]
direction_map = {
    "relax": Key.right,
    "concentrating": Key.right,
    "memory": Key.down,
    "stress": Key.left,
}
keyboard = Controller()

# === 單筆預測並模擬按鍵 ===

N_EEG = 0
N_manual = 0

def predict_and_press(path):
    global N_EEG, N_manual

    try:
        x_raw    = load_single_txt_with_features(path)
        x_scaled = scaler.transform(x_raw.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(x_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            signal   = label_map[pred_idx]

        print(f"✅ 預測方向：{signal}")

        # ---- 決定使用 EEG 還是手動 ----
        choice = input("▶ 直接採用預測(Enter) 或輸入 w/a/s/d 修正方向？ ").strip().lower()

        # ⇢ 若 choice 為空字串 → 採用預測方向
        if choice == "":
            direction_key = direction_map[signal]
            is_eeg_move = True
        else:
            # 手動指定 w/a/s/d → 對應四方向
            manual_dir = {
                "w": Key.up,
                "s": Key.down,
                "a": Key.left,
                "d": Key.right
            }.get(choice)

            if manual_dir is None:
                print("⚠️ 無效輸入，跳過本次移動")
                return
            direction_key = manual_dir
            is_eeg_move   = False

        # ---- 輸入步數 ----
        try:
            steps = int(input("🧭 請輸入步數(整數)："))
        except ValueError:
            print("⚠️ 輸入無效，預設 1 步")
            steps = 1
        if steps <= 0:
            print("⚠️ 步數需 > 0，取消本次")
            return

        # ---- 執行鍵盤輸入 ----
        focus_maze_window()
        for _ in range(steps):
            keyboard.press(direction_key)
            time.sleep(0.07)
            keyboard.release(direction_key)
        focus_terminal_window()

        # ---- 更新計數 ----
        if is_eeg_move:
            N_EEG += steps
        else:
            N_manual += steps

        # ---- 顯示統計 ----
        print(f"📊 N_EEG = {N_EEG}, N_manual = {N_manual}")

    except Exception as e:
        print(f"⚠️ 預測/操作失敗：{e}")
# === 主執行迴圈 ===

def main():
    print("🧠 進入即時預測模式，0.5 秒輪詢一次 …")
    path = "../data/bci_dataset_113-2/S01/6.txt"
    last_mtime = None           # 上次看到的檔案修改時間

    while True:
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)   # 取得最後修改時間 (float, 秒)
            except OSError as e:
                # 檔案正被寫入而暫時鎖住 → 稍後再試
                print(f"⚠️ 讀取 {path} 失敗：{e}")
                time.sleep(0.5)
                continue

            # 只有時間戳不同才重新推論
            if mtime != last_mtime:
                last_mtime = mtime
                predict_and_press(path)

        else:
            print("⚠️ 找不到檔案，等待中…")

        time.sleep(0.5)         # 每半秒輪詢一次

if __name__ == "__main__":
    main()
