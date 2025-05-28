import tkinter as tk
import random
import time
from threading import Thread

GRID_SIZE = 4
START_LENGTH = 4  # 初始關卡的亮燈數

class LightGame:
    def __init__(self, root):
        self.root = root
        self.root.title("亮燈遊戲 - 關卡制")
        self.buttons = []
        self.sequence = []
        self.user_sequence = []
        self.lock_input = True
        self.level = 1

        self.build_grid()
        self.info_label = tk.Label(root, text="按下『開始遊戲』來挑戰", font=("Arial", 14))
        self.info_label.grid(row=GRID_SIZE, column=0, columnspan=GRID_SIZE)

        self.start_button = tk.Button(root, text="開始遊戲", command=self.start_game)
        self.start_button.grid(row=GRID_SIZE+1, column=0, columnspan=GRID_SIZE, sticky="we")

    def build_grid(self):
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                btn = tk.Button(self.root, width=8, height=4,
                                command=lambda i=i, j=j: self.on_click(i, j))
                btn.grid(row=i, column=j)
                row.append(btn)
            self.buttons.append(row)

    def flash_sequence(self):
        self.lock_input = True
        seq_len = START_LENGTH + self.level - 1
        self.sequence = [ (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)) for _ in range(seq_len) ]
        self.info_label.config(text=f"第 {self.level} 關：記住 {seq_len} 個格子！")

        time.sleep(1)  # 給玩家一點時間準備

        for i, j in self.sequence:
            self.buttons[i][j].config(bg="yellow")
            self.root.update()
            time.sleep(0.5)
            self.buttons[i][j].config(bg="SystemButtonFace")
            self.root.update()
            time.sleep(0.2)

        self.user_sequence = []
        self.lock_input = False

    def start_game(self):
        self.level = 1
        Thread(target=self.flash_sequence).start()

    def next_level(self):
        self.level += 1
        Thread(target=self.flash_sequence).start()

    def on_click(self, i, j):
        if self.lock_input:
            return
        self.user_sequence.append((i, j))
        idx = len(self.user_sequence) - 1
        if self.user_sequence[idx] != self.sequence[idx]:
            self.show_result("失敗！遊戲結束", restart=True)
            self.lock_input = True
        elif len(self.user_sequence) == len(self.sequence):
            self.info_label.config(text=f"通過第 {self.level} 關！準備下一關...")
            self.lock_input = True
            self.root.after(1000, self.next_level)

    def show_result(self, msg, restart=False):
        popup = tk.Toplevel()
        popup.title("結果")
        tk.Label(popup, text=msg, font=("Arial", 16)).pack(padx=20, pady=20)

        if restart:
            tk.Button(popup, text="重新開始", command=lambda: [popup.destroy(), self.start_game()]).pack(pady=10)
        else:
            tk.Button(popup, text="關閉", command=popup.destroy).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    game = LightGame(root)
    root.mainloop()
