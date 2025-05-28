#Requires AutoHotkey v2.0

; ========== 可調參數 ==========
EEGWin     := "Biopac Student Lab"
TargetFile := "\LAPTOP-Amy\BIOPAC\data\data.txt"
TxtTitle   := "data.txt"
RecordSec  := 10               ; 錄製秒數
IntervalSec := 12              ; 每隔幾秒執行一次錄製
; ==============================

global paused := false

SetTimer(RecordAndSave, IntervalSec * 1000)  ; 每隔 IntervalSec 秒執行一次

RecordAndSave() {
    global paused
    if paused {
        ToolTip("⏸️ 已暫停")
        return
    }

    try {
        DetectHiddenWindows(True)
        if !WinActivate(EEGWin)
            throw Error("找不到 EEG 軟體視窗: " EEGWin)
        WinWaitActive(EEGWin,,3)
        Sleep 200
        CoordMode("Mouse", "Window")
        WinGetPos(&x,&y,&w,&h, EEGWin)
        MouseClick("Left", x + w/2, y + h/2, 1, 0)
        Sleep 150

        Send("^ ")
        Sleep RecordSec 1000
        Send("^ ")

        Sleep 500
        Send("^a")
        Sleep 200
        Send("^l")
        Sleep 500
        if (A_Clipboard = "")
            throw Error("剪貼簿為空，可能複製失敗")

        Run(Format('notepad.exe "{}"', TargetFile))
        if !WinWaitActive(TxtTitle " - Notepad",,5)
            throw Error("未能開啟 " TargetFile)

        Sleep 300
        Send("^a{Del}")
        Sleep 150
        Send("^v")
        Sleep 300
        Send("^s")
        Sleep 300
        WinClose("A")
        ToolTip("✅ 成功寫入資料 " . TargetFile, 0, 0)
        Sleep 1000
        ToolTip()
    } catch e {
        MsgBox("⚠️ 錯誤發生: " . e.Message)
    }
}

Enter:: {
    global paused
    paused := !paused
    if paused {
        ToolTip("⏸️ 暫停中（按 Enter 繼續）", 0, 0)
    } else {
        ToolTip("▶️ 已恢復", 0, 0)
        Sleep 1000
        ToolTip()
    }
}

Esc::ExitApp