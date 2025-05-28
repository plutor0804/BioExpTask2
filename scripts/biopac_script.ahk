#Requires AutoHotkey v2.0
SetTitleMatchMode "RegEx"

running := false
paused := false
EEGWin := "Biopac Student Lab"
target_file := "\\LAPTOP-Amy\BIOPAC\data\data.txt"

if !WinExist(EEGWin)
    return
WinActivate EEGWin

MainLoop() {
    global running, paused, EEGWin, target_file
    running := true
    MsgBox "開始主迴圈！再次按 Ctrl+Enter 可暫停，Ctrl+Esc 可停止。"

    while running {
        if paused {
            Sleep 500
            continue
        }

        if !WinExist(EEGWin)
            return
        WinActivate EEGWin

        ; Step 1: 觸發波形擷取
        Send "^ "  ; Ctrl + Space
        Sleep 20000
        Send "^ "  ; Ctrl + Space again

        ; Step 2: 複製波形資料
        Sleep 200
        Clipboard := ""  ; 清空剪貼簿
        Send "^a"
        Sleep 100
        Send "^l"
        Sleep 2000
        if !ClipWait(2)
            return

        ; Step 3: 開啟目標檔案並貼上
        Run target_file
        WinWaitActive "data.txt"
        Sleep 500
        Send "^a"       ; 全選
        Sleep 100
        Send "^v"       ; 貼上新資料
        Sleep 100
        Send "^s"       ; 儲存
        Sleep 200
        Send "!{F4}"    ; 關閉檔案視窗

        ; Step 4: 返回 EEG 軟體並清除畫面
        WinActivate EEGWin
        Send "^x"       ; 清除畫面
        Sleep 3000
    }
}

^Enter:: {
    global running, paused  
    if !running {
        MainLoop()
    } else {
        paused := !paused
        ToolTip paused ? "⏸ 已暫停" : "▶️ 恢復執行"
        SetTimer () => ToolTip(), -1000
    }
}

^Esc:: {
    global running
    running := false
    MsgBox "腳本已停止！"
    ExitApp
}
