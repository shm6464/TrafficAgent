# 交通运维知识库问答系统 - 停止服务脚本（PowerShell）
#
# 用法：在项目根目录执行
#     .\scripts\stop.ps1
#
# 停止占用 8000 端口（或指定端口）的 uvicorn 服务。

param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "正在停止 FastAPI 服务（端口 $Port）..." -ForegroundColor Cyan

# 查找监听指定端口的进程（uvicorn / python）
$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $conns) {
    Write-Host "端口 $Port 没有正在监听的进程，无需停止。" -ForegroundColor Gray
    exit 0
}

$pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pid in $pids) {
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "停止进程 $($proc.ProcessName) (PID $pid)..." -ForegroundColor Yellow
        Stop-Process -Id $pid -Force
    }
}

Write-Host "已停止。重新启动请执行 .\scripts\start.ps1" -ForegroundColor Green
