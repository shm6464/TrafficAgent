# 交通运维知识库问答系统 - 一键启动脚本（PowerShell）
#
# 本地直接启动 FastAPI 服务（不起 Docker，不构建镜像）。
# 用法：在项目根目录执行
#     .\scripts\start.ps1
# 或指定端口：
#     .\scripts\start.ps1 -Port 8080
#
# 停止：Ctrl+C（前台进程），或 .\scripts\stop.ps1

param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 交通运维知识库问答系统 - FastAPI 服务" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 检查虚拟环境
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "未找到虚拟环境 $Python，请先完成环境初始化。"
    exit 1
}

# 检查 .env
if (-not (Test-Path (Join-Path $Root ".env"))) {
    Write-Warning "未找到 .env，模型端点可能未配置。"
}

Write-Host "启动服务：http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "健康检查：Invoke-RestMethod http://127.0.0.1:$Port/healthz" -ForegroundColor Gray
Write-Host "接口文档：http://127.0.0.1:$Port/docs" -ForegroundColor Gray
Write-Host ""

# 前台启动 uvicorn（单 worker，避免重复加载模型）
& $Python -m uvicorn api.main:app --host 127.0.0.1 --port $Port
