from pynput.keyboard import Controller, Key
from pynput import keyboard as kb
import time
import random


keyboard = Controller()
simulation_active = False
predicted_signal = None

def press_key(key):
    keyboard.press(key)
    time.sleep(1.2)
    keyboard.release(key)

def on_press(key):
    global simulation_active, predicted_signal

    try:
        k = key.char.lower()
        if k == 'i':
            predicted_signal = "relax"
        elif k == 'j':
            predicted_signal = "memory"
        elif k == 'k':
            predicted_signal = "concentrating"
        elif k == 'l':
            predicted_signal = "stress"
    except AttributeError:
        if key == Key.space:
            if not simulation_active:
                simulation_active = True
                print("✅ 模擬啟動！可按 R (relax), M (memory), C (concentrating), S (stress)")
            else:
                print("⚠️ 模擬已啟動，不需重複按 Space。")
        elif key == Key.esc:
            print("🛑 模擬結束，再見！")
            return False  # 停止 listener

def simulation_loop():
    global predicted_signal

    print("💡 請按下空白鍵開始模擬，ESC 結束")
    while True:
        if not simulation_active:
            time.sleep(0.1)
            continue

        if predicted_signal:
            print(f"[EEG 模擬] 狀態分類：{predicted_signal}")
            if predicted_signal == "relax":
                press_key(Key.up)
            elif predicted_signal == "concentrating":
                press_key(Key.down)
            elif predicted_signal == "memory":
                press_key(Key.left)
            elif predicted_signal == "stress":
                press_key(Key.right)

            predicted_signal = None  # 重設等待下次輸入
        # time.sleep(0.1)

if __name__ == "__main__":
    with kb.Listener(on_press=on_press) as listener:
        simulation_loop()
        listener.join()
