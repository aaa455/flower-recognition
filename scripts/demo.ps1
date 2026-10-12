param(
    [string]$Checkpoint = 'runs/experiment-01/best.pt',
    [string]$Image = 'data/flower_photos/daisy/100080576_f52e8ee070_n.jpg',
    [string]$Output,
    [ValidateSet('cpu', 'cuda', 'auto')]
    [string]$Device = 'cpu'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'

function Invoke-ProjectPython {
    param([string[]]$ArgumentList)
    & $ProjectPython -X utf8 @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令执行失败，退出码：$LASTEXITCODE"
    }
}

# 参数中的相对路径统一从仓库根目录解析，结束后恢复调用者的目录。
Push-Location -LiteralPath $ProjectRoot
try {
    foreach ($RequiredFile in @($ProjectPython, $Checkpoint, $Image)) {
        if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
            throw "找不到文件：$RequiredFile。请先按 README 准备环境、模型和图片。"
        }
    }
    if (-not $Output) {
        $RunName = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
        $Output = "runs/demo/$RunName/prediction.json"
    }
    if (Test-Path -LiteralPath $Output) {
        throw "输出路径已存在：$Output。请换一个 -Output，避免覆盖已有结果。"
    }

    Write-Host '步骤 1/2：检查网络结构（随机输入，仅展示张量形状）。'
    Invoke-ProjectPython -ArgumentList @('-m', 'src.model')
    Write-Host '步骤 2/2：加载已训练的模型，预测指定图片。'
    Invoke-ProjectPython -ArgumentList @('-m', 'src.predict', '--checkpoint', $Checkpoint,
        '--image', $Image, '--output', $Output, '--device', $Device)
    Write-Host "演示完成，预测记录：$Output"
}
finally {
    Pop-Location
}
