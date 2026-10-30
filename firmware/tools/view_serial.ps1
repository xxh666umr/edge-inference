# 实时查看 STM32F103 板子串口输出（USART1，115200 8N1）
# 自动识别 CH340 对应的 COM 端口；按 Ctrl+C 或关闭窗口退出。
# 由同目录的「查看串口输出.bat」双击调用，也可以手动：
#   powershell -ExecutionPolicy Bypass -File firmware\tools\view_serial.ps1

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$ErrorActionPreference = "Stop"

# 1) 自动找 CH340（USB 转串口）对应的 COM 口
$port = $null
$devs = Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'CH340|USB-SERIAL|USB Serial' }
foreach ($d in $devs) {
    if ($d.Name -match '\((COM\d+)\)') { $port = $Matches[1]; break }
}
if (-not $port) {
    $cands = [System.IO.Ports.SerialPort]::GetPortNames()
    if ($cands.Count -eq 1) { $port = $cands[0] }
}
if (-not $port) {
    Write-Host "没有找到串口设备。" -ForegroundColor Red
    Write-Host "请检查：板子 USB 线是否插好？板子是否上电？" -ForegroundColor Yellow
    Write-Host ("当前系统串口: " + (([System.IO.Ports.SerialPort]::GetPortNames()) -join ', '))
    Read-Host "按回车退出"
    exit 1
}

# 2) 打开串口，持续打印
$sp = New-Object System.IO.Ports.SerialPort($port, 115200,
        [System.IO.Ports.Parity]::None, 8, [System.IO.Ports.StopBits]::One)
$sp.Encoding = [System.Text.Encoding]::UTF8
$sp.NewLine = "`r`n"
try {
    $sp.Open()
} catch {
    Write-Host ("打开 " + $port + " 失败：" + $_.Exception.Message) -ForegroundColor Red
    Write-Host "可能被别的串口软件占用了，先关掉它再试。" -ForegroundColor Yellow
    Read-Host "按回车退出"
    exit 1
}

try { Clear-Host } catch { }
Write-Host ("已连接 " + $port + " @ 115200    板子每 5 秒打印一帧，按 Ctrl+C 退出") -ForegroundColor Green
Write-Host ("-" * 72)
$buf = ""
while ($true) {
    try {
        $buf += $sp.ReadExisting()
        if ($buf.Length -gt 0) {
            # 一帧以 "====" 结尾，简单起见整段打印（去掉多余空行）
            $text = $buf.Replace("`r", "")
            foreach ($line in ($text -split "`n")) { if ($line.Trim()) { Write-Host $line } }
            $buf = ""
        }
        Start-Sleep -Milliseconds 200
    } catch {
        Write-Host ("串口读取出错（板子被拔掉？）：" + $_.Exception.Message) -ForegroundColor Red
        break
    }
}
$sp.Close()
