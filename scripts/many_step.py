from pynput.keyboard import Controller, Key
from pynput import keyboard as kb
import pyautogui
import time

keyboard = Controller()
simulation_active = False
predicted_signal = None
cancel_requested = False

# 遊戲瀏覽器分頁大約位置（需你自行測量）
CLICK_MAZE_X = 581
CLICK_MAZE_Y = 30

# Terminal 視窗大約位置（需你自行測量）
CLICK_TERMINAL_X = 1271
CLICK_TERMINAL_Y = 978

def press_key(key, steps=1):
    for i in range(steps):
        keyboard.press(key)
        time.sleep(0.07) # 不要動!!
        keyboard.release(key)
        print(f"  ↳ 第 {i+1} 步完成")

def press_key_new(key, steps=1):
    keyboard.press(key)
    time.sleep(0.07*steps) # 不要動!!
    keyboard.release(key)
    print(f"  ↳ 完成")

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
    global simulation_active, predicted_signal, cancel_requested

    try:
        k = key.char.lower()
        direction_map = {
            'i': 'relax',
            'j': 'memory',
            'k': 'concentrating',
            'l': 'stress'
        }
        if k in direction_map:
            predicted_signal = direction_map[k]
            cancel_requested = False

    except AttributeError:
        if key == Key.space:
            if not simulation_active:
                simulation_active = True
                print("✅ 模擬啟動！可按 I (relax), J (memory), K (concentrating), L (stress)")
            else:
                print("⚠️ 模擬已啟動，不需重複按 Space。")
        elif key == Key.esc:
            print("🛑 模擬結束，再見！")
            return False

def simulation_loop():
    global predicted_signal, cancel_requested

    print("💡 請按下空白鍵開始模擬，ESC 結束")
    while True:
        if not simulation_active:
            # time.sleep(0.1) # 不確定拿掉可不可以
            continue

        if predicted_signal:
            print(f"[EEG 模擬] 狀態分類：{predicted_signal}")

            try:
                steps = int(input("🧭 請輸入要走幾步（數字）："))
                if steps < 1:
                    print("🚫 本次方向輸入已取消。")
                    predicted_signal = None
                    continue
            except ValueError:
                print("⚠️ 輸入無效，將預設為 1 步。")
                steps = 1

            direction_key_map = {
                "relax": Key.up,
                "concentrating": Key.down,
                "memory": Key.left,
                "stress": Key.right
            }

            if predicted_signal in direction_key_map:
                focus_maze_window()
                # time.sleep(0.3) #人工按壓buffer，成功了可以拿掉
                press_key(direction_key_map[predicted_signal], steps)
                focus_terminal_window()  # 走完步數後回 Terminal

            predicted_signal = None

if __name__ == "__main__":
    with kb.Listener(on_press=on_press) as listener:
        simulation_loop()
