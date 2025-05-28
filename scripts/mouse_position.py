import pyautogui
import time

print("請把滑鼠移到你想知道座標的地方，程式會每秒印一次位置，按 Ctrl+C 停止")

try:
    while True:
        x, y = pyautogui.position()
        print(f"滑鼠座標：X={x}, Y={y}")
        time.sleep(1)
except KeyboardInterrupt:
    print("已結束座標監測")
