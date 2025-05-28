import os
import time
import torch
import numpy as np
import joblib
import pyautogui
import threading
from pynput.keyboard import Controller, Key, Listener

# === 遊戲視窗位置（需手動測量）===
CLICK_MAZE_X = 581
CLICK_MAZE_Y = 30
CLICK_TERMINAL_X = 1438
CLICK_TERMINAL_Y = 977

N_manual = 0
N_EEG = 0

def focus_maze_window():
    try:
        pyautogui.click(CLICK_MAZE_X, CLICK_MAZE_Y)
        print("🔄 切換到迷宮遊戲視窗")
    except Exception as e:
        print(f"❌ 迷宮視窗切換失敗：{e}")

def focus_terminal_window():
    try:
        pyautogui.click(CLICK_TERMINAL_X, CLICK_TERMINAL_Y)
        print("🔄 切換回 Terminal 視窗")
    except Exception as e:
        print(f"❌ Terminal 切換失敗：{e}")

# === 模型架構 ===
class EEGMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# === 特徵擷取 ===
def extract_band_power(vec, freqs):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "low-beta": (13, 20),
        "high-beta": (20,30),
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
    vec = np.loadtxt(txt_path)
    vec = vec[:10000]
    fft_result = np.fft.rfft(vec, n=10000)
    power = np.abs(fft_result) ** 2
    log_power = np.log1p(power)
    freqs = np.fft.rfftfreq(10000, d=1/500)
    stat_feat = extract_features(log_power)
    band_feat = extract_band_power(log_power, freqs)
    return np.concatenate([log_power, stat_feat, band_feat])

# === 載入模型與 scaler ===
model = EEGMLP(5019, 4)
model.load_state_dict(torch.load("../models/subject9_best_model_1.pt", weights_only=True))
model.eval()
scaler = joblib.load("../models/scaler_subject9.pkl")

label_map = ["relax", "concentrating", "stress", "memory"]
signal_to_key = {
    "relax": Key.up,
    "concentrating": Key.down,
    "memory": Key.left,
    "stress": Key.right
}
keyboard = Controller()

# === 腦波預測控制 ===
def predict_and_press(path="live_data.txt"):
    global N_EEG
    try:
        x_raw = load_single_txt_with_features(path)
        x_scaled = scaler.transform(x_raw.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            output = model(x_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            signal = label_map[pred_idx]

        print(f"✅ 腦波預測結果：{signal}")

        if signal in signal_to_key:
            try:
                steps = int(input("🧭 請輸入步數（整數）："))
            except ValueError:
                print("⚠️ 預設步數為 1")    
                steps = 1

            focus_maze_window()
            for _ in range(steps):
                keyboard.press(signal_to_key[signal])
                time.sleep(0.07)
                keyboard.release(signal_to_key[signal])
            focus_terminal_window()

            N_EEG += steps
            print(f"📊 EEG 控制總步數：{N_EEG}")

    except Exception as e:
        print(f"⚠️ EEG 預測錯誤：{e}")

# === 手動控制監聽器 ===
manual_key_map = {
    'i': "relax",        # ↑
    'k': "concentrating",# ↓
    'l': "stress",       # →
    'j': "memory"        # ←
}

def on_press(key):
    global N_manual
    try:
        k = key.char.lower()
        if k in manual_key_map:
            signal = manual_key_map[k]
            if signal in signal_to_key:
                try:
                    steps = int(input(f"🧭 [手動] 請輸入 {signal} 的步數（整數）："))
                except ValueError:
                    print("⚠️ 無效輸入，預設步數為 1")
                    steps = 1

                focus_maze_window()
                for _ in range(steps):
                    keyboard.press(signal_to_key[signal])
                    time.sleep(0.07)
                    keyboard.release(signal_to_key[signal])
                focus_terminal_window()

                N_manual += steps
                print(f"🕹️ 手動控制方向：{signal}，總手動步數：{N_manual}")
    except AttributeError:
        pass  # 忽略特殊鍵


def start_manual_listener():
    listener = Listener(on_press=on_press)
    listener.start()

# === 主迴圈 ===
def main():
    print("🧠 EEG 即時預測啟動，每 20 秒掃描一次資料")
    print("🎮 手動控制：i=↑ j=← k=↓ l=→")
    start_manual_listener()

    while True:
        if os.path.exists("../data/bci_dataset_113-2/S01/1.txt"):
            predict_and_press("../data/bci_dataset_113-2/S01/1.txt")
        else:
            print("⚠️ 找不到 EEG 檔案，等待中...")
        time.sleep(20)

if __name__ == "__main__":
    main()
