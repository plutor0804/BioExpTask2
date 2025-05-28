import os, time, numpy as np, torch, joblib, pyautogui
from pynput.keyboard import Controller, Key
from scipy.signal import butter, filtfilt

# ========== 參數 ==========
# ---- EEG 特徵 ----
VEC_DIM   = 248          # 496 (=10 s) 或 251 (=5 s)
META_DIM  = 20           # 12+5+3
NUM_CLASS = 4

# ---- 時間片長度 ----
SEG_LEN   = 2500 #if VEC_DIM == 496 else 2500   # 5000 點=10 s；2500 點=5 s
CENTER_EXTRACT = True                          # 取正中央一段

# ---- 檔案 & 模型 ----
MODEL_PATH  = "../models/5s model/10.pt"
SCALER_PATH = "../models/5s model/10.pkl"
EEG_FILE    = "../data/data_val/stress/1.txt"           # 監聽的 txt 檔

# ---- 遊戲座標 ----
CLICK_MAZE_X, CLICK_MAZE_Y      = 357,  511
CLICK_TERM_X, CLICK_TERM_Y      = 1734, 1005

# ========== UI 工具 ==========
def focus_maze():   pyautogui.click(CLICK_MAZE_X, CLICK_MAZE_Y)
def focus_term():   pyautogui.click(CLICK_TERM_X, CLICK_TERM_Y)

# ========== 模型 ==========
class HybridCNN(torch.nn.Module):
    def __init__(self, vec_dim, meta_dim, num_cls):
        super().__init__()
        self.vec_dim = vec_dim
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, 5, padding=2), torch.nn.BatchNorm1d(32), torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(32, 64, 5, padding=2), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(meta_dim, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64+64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_cls)
        )



    def forward(self, x):
        v   = x[:, :self.vec_dim].unsqueeze(1)
        met = x[:, self.vec_dim:]
        return self.classifier(torch.cat([self.cnn(v), self.mlp(met)], 1))

# ========== 特徵 ==========
def extract_features(vec):
    v, vs = vec, vec + 1e-8
    f = [np.mean(v), np.std(v), np.sqrt(np.mean(v**2)), np.min(v), np.max(v),
         np.percentile(v,25), np.median(v), np.percentile(v,75),
         np.mean((v-np.mean(v))**3)/(np.std(v)**3+1e-8),
         np.mean((v-np.mean(v))**4)/(np.std(v)**4+1e-8),
         np.sum(v**2),
         -np.sum((vs/vs.sum())*np.log(vs/vs.sum()))]
    return np.array(f)

def extract_band_power(vec, freqs):
    bands = {"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,50)}
    return np.array([np.mean(vec[(freqs>=lo)&(freqs<hi)]) for lo,hi in bands.values()])

def compute_ratios(b):
    θ, α, β = b[1], b[2], b[3]
    return np.array([θ/(α+1e-8), α/(β+1e-8), β/(θ+1e-8)])

# ---- 濾波器一次建立 ----
b_bp, a_bp = butter(4, [0.5,50], btype='bandpass', fs=500)

def load_single_txt(txt_path):

    with open(txt_path, encoding='utf-8') as f:
        raw = np.loadtxt(f, skiprows=1)
    if len(raw) < SEG_LEN:
        raise ValueError("檔案點數不足")

    # ---- 中央 SEG_LEN ----
    s = len(raw)//2 - SEG_LEN//2
    seg = raw[s:s+SEG_LEN]

    # ---- 濾波 & FFT ----
    filt  = filtfilt(b_bp, a_bp, seg)
    power = np.abs(np.fft.rfft(filt, n=SEG_LEN))**2
    logp  = np.log1p(power)

    freqs = np.fft.rfftfreq(SEG_LEN, d=1/500)
    idx   = (freqs >= 0.5) & (freqs <= 50)

    vec    = logp[idx]          # 248 bin
    f_used = freqs[idx]         # 248 freq


    # ---- 特徵 ----
    stat  = extract_features(vec)
    band  = extract_band_power(vec, f_used)   # 兩者長度相同
    ratio = compute_ratios(band)
    return np.concatenate([vec, stat, band, ratio])

# ========== 載入模型 ==========
model  = HybridCNN(VEC_DIM, META_DIM, NUM_CLASS)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
scaler = joblib.load(SCALER_PATH)

label_map = ["relax", "concentrating", "stress", "memory"]
dir_map   = {"relax":Key.right, "concentrating":Key.right, "stress":Key.left, "memory":Key.down}
kbd = Controller()

N_EEG = N_manual = 0

def predict_and_press(p):
    global N_EEG, N_manual
    x = scaler.transform(load_single_txt(p).reshape(1,-1))
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32))
        label = label_map[pred.argmax(1).item()]
    print("✅ 預測：", label)

    choice = input("▶ Enter=採用｜w/a/s/d=").strip().lower()
    if choice in ['w','a','s','d']:
        key = {'w':Key.up,'s':Key.down,'a':Key.left,'d':Key.right}[choice]
        is_eeg=False
    else:
        key = dir_map[label]; is_eeg=True

    try: steps=int(input("🧭 步數："))
    except: steps=1
    if steps<=0: return

    focus_maze()
    for _ in range(steps):
        kbd.press(key); time.sleep(0.07); kbd.release(key)
    focus_term()

    if is_eeg: N_EEG += steps
    else:      N_manual += steps
    print(f"📊 EEG:{N_EEG} | 手動:{N_manual}")

# ========== 監聽迴圈 ==========
def main():
    print("🧠 即時預測中… (檔案變動觸發)"); last_m = None
    while True:
        if os.path.exists(EEG_FILE):
            m = os.path.getmtime(EEG_FILE)
            if m != last_m:
                last_m = m
                try:    predict_and_press(EEG_FILE)
                except Exception as e: print("⚠️", e)
        else:
            print("⚠️ 找不到檔案", EEG_FILE)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
