import torch
import numpy as np
import joblib
import time
from pynput.keyboard import Controller, Key
from pynput import keyboard as kb
import pyautogui
import glob
import os

# === 🧠 特徵擷取函數 ===
def extract_band_power(vec, freqs):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "low-beta": (13, 20),
        "high-beta": (20, 30),
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
        np.mean(vec),                     # 平均值
        np.std(vec),                      # 標準差
        np.sqrt(np.mean(vec ** 2)),       # 均方根
        np.min(vec),                      # 最小值
        np.max(vec),                      # 最大值
        np.percentile(vec, 25),           # 第25百分位
        np.median(vec),                   # 中位數
        np.percentile(vec, 75),           # 第75百分位
        np.mean((vec - np.mean(vec))**3) / (np.std(vec)**3 + 1e-8),  # 偏度
        np.mean((vec - np.mean(vec))**4) / (np.std(vec)**4 + 1e-8),  # 峰度
        np.sum(vec**2),                   # 能量
        -np.sum((vec_safe/np.sum(vec_safe)) * np.log(vec_safe/np.sum(vec_safe)))  # 熵
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

# =========================================
# 🧠 模型定義（需與訓練階段一致）
# =========================================
class EEGMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EEGMLP, self).__init__()
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

# =========================================
# 📥 載入模型與標準化器
# =========================================
label_map = ["relax", "concentrating", "stress", "memory"]

model = EEGMLP(5019, 4)
model.load_state_dict(torch.load("subject9_best_model_1.pt"))
model.eval()
scaler = joblib.load("scaler_subject9.pkl")

# =========================================
# 🎮 鍵盤模擬控制
# =========================================
keyboard = Controller()
simulation_active = False

# 遊戲瀏覽器分頁大約位置（需你自行測量）
CLICK_MAZE_X = 432
CLICK_MAZE_Y = 24

# Terminal 視窗大約位置（需你自行測量）
CLICK_TERMINAL_X = 1271
CLICK_TERMINAL_Y = 978
# =========================================
# ⌨️ 鍵盤監聽行為定義
# =========================================
def press_key(key):
    keyboard.press(key)
    time.sleep(0.07) # 不要動!!
    keyboard.release(key)

def focus_maze_window():
    try:
        pyautogui.click(CLICK_MAZE_X, CLICK_MAZE_Y)
        print("🔄 用滑鼠點擊切換到迷宮遊戲視窗")
        # time.sleep(0.5)
    except Exception as e:
        print(f"❌ 滑鼠點擊切換迷宮視窗失敗：{e}")

def focus_terminal_window():
    try:
        pyautogui.click(CLICK_TERMINAL_X, CLICK_TERMINAL_Y)
        print("🔄 用滑鼠點擊切換到 Terminal 視窗")
        # time.sleep(0.5)
    except Exception as e:
        print(f"❌ 滑鼠點擊切換 Terminal 視窗失敗：{e}")

def on_press(key):
    global simulation_active
    try:
        k = key.char.lower()
        if k in ['i', 'k', 'j', 'l']:
            signal_map = {
                'i': 'relax',
                'k': 'concentrating',
                'l': 'stress',
                'j': 'memory'
            }

            direction_map = {
                "relax": ("↑", Key.up),
                "concentrating": ("↓", Key.down),
                "memory": ("←", Key.left),
                "stress": ("→", Key.right)
            }

            signal = signal_map[k]
            direction_arrow, key_to_press = direction_map[signal]

            try:
                step_count = int(input(f"🖐️ 手動模式 → {signal} → {direction_arrow}，請輸入步數："))
                if step_count <= 0:
                    print("🚫 本次方向輸入已取消。")
                    return
            except ValueError:
                print("⚠️ 無效輸入，預設步數為 1。")
                step_count = 1

            print(f"➡️ 執行方向 {direction_arrow}，共 {step_count} 步")
            # time.sleep(0.3) # 為了給我切分頁預留的delay
            focus_maze_window()
            for _ in range(step_count):
                press_key(key_to_press)
                # time.sleep(0.5)  # 每次按鍵間的延遲
            focus_terminal_window()

        elif k == 'n':
            print("🔍 開始對 6 個資料檔進行一次性預測...")

            direction_map = {
                "relax": ("↑", Key.up),
                "concentrating": ("↓", Key.down),
                "memory": ("←", Key.left),
                "stress": ("→", Key.right)
            }

            for i in range(1, 7):
                file_path = f"data/{i}-8.txt"
                try:
                    x_raw = load_single_txt_with_features(file_path)
                    x_scaled = scaler.transform(x_raw.reshape(1, -1))
                    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

                    with torch.no_grad():
                        output = model(x_tensor)
                        pred_idx = torch.argmax(output, dim=1).item()
                        signal = label_map[pred_idx]

                        if signal in direction_map:
                            direction_arrow, key_to_press = direction_map[signal]
                            
                            # ➕ 每一筆資料都詢問使用者步數
                            try:
                                step_count = int(input(f"📁 {file_path} → 預測: {signal} → {direction_arrow}，請輸入步數："))
                                if step_count <= 0:
                                    print("🚫 本次方向輸入已取消，等待下次指令。")
                                    continue
                            except ValueError:
                                print("⚠️ 無效輸入，預設步數為 1。")
                                step_count = 1

                            print(f"➡️ 執行方向 {direction_arrow}，共 {step_count} 步")
                            # time.sleep(0.3) # 為了給我切分頁預留的delay
                            focus_maze_window()
                            for _ in range(step_count):
                                press_key(key_to_press)
                                # time.sleep(0.5)  # 每次按鍵間的延遲
                            focus_terminal_window()

                    # time.sleep(0.5)

                except Exception as e:
                    print(f"❌ 錯誤：無法處理 {file_path}，原因：{e}")

    except AttributeError:
        if key == Key.space:
            if not simulation_active:
                simulation_active = True
                print("✅ 模擬啟動！")
            else:
                print("⚠️ 模擬已經啟動，無需再次按空白鍵。")
        elif key == Key.esc:
            print("🛑 模擬結束，再見！")
            return False

# =========================================
# 🔁 模擬主迴圈
# =========================================
def simulation_loop():
    print("💡 請按下空白鍵啟動模擬，按 ESC 結束程式")
    while True:
        if not simulation_active:
            # time.sleep(0.1)
            continue
        # time.sleep(0.1)

# =========================================
# ▶️ 啟動鍵盤監聽與模擬迴圈
# =========================================
with kb.Listener(on_press=on_press) as listener:
    simulation_loop()
    listener.join()
