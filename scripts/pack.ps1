# PowerShell 打包脚本 - 钢铁表面缺陷检测系统
# 排除 .venv, node_modules, logs, runs, data/tmp 等冗余文件

$source = "f:\steel-defect-detection"
$zipPath = "$source\steel-defect-detection.zip"
$tempDir = "$source\temp_pack"

Write-Host "=== 开始打包项目 ===" -ForegroundColor Cyan

if (Test-Path $zipPath) {
    Write-Host "清理旧的压缩包..."
    Remove-Item $zipPath -Force
}
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}

New-Item -ItemType Directory -Path $tempDir | Out-Null

# 定义需要复制的文件和文件夹
$itemsToCopy = @(
    "src", "docs", "models", "scripts", "tests",
    "frontend/dist",
    "app.py", "server.py", "main.py", "cli.py", "config.yaml", "requirements.txt",
    "setup.bat", "start.bat", "start.ps1", "pytest.ini", "read_docx_xlsx.py",
    "yolov8n.pt", "yolo26n.pt", "images.zip", "labels.zip", ".env.example"
)

foreach ($item in $itemsToCopy) {
    $srcPath = Join-Path $source $item
    $destPath = Join-Path $tempDir $item
    if (Test-Path $srcPath) {
        $parent = Split-Path $destPath
        if (!(Test-Path $parent)) {
            New-Item -ItemType Directory -Path $parent | Out-Null
        }
        Write-Host "正在复制: $item"
        Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
    } else {
        Write-Host "跳过不存在的项: $item" -ForegroundColor Yellow
    }
}

# 创建必要但为空的文件夹结构
Write-Host "创建数据空文件夹结构..."
New-Item -ItemType Directory -Path "$tempDir\data" | Out-Null
New-Item -ItemType Directory -Path "$tempDir\data\images" | Out-Null
New-Item -ItemType Directory -Path "$tempDir\data\uploads" | Out-Null
New-Item -ItemType Directory -Path "$tempDir\data\exports" | Out-Null
New-Item -ItemType Directory -Path "$tempDir\logs" | Out-Null

# 压缩临时文件夹
Write-Host "正在生成压缩包: $zipPath ..." -ForegroundColor Green
Compress-Archive -Path "$tempDir\*" -DestinationPath $zipPath -Force

# 清理临时目录
Write-Host "清理临时文件..."
Remove-Item $tempDir -Recurse -Force

if (Test-Path $zipPath) {
    $size = (Get-Item $zipPath).Length / 1MB
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "打包成功！" -ForegroundColor Green
    Write-Host "压缩包大小: $($size.ToString('F2')) MB" -ForegroundColor Green
    Write-Host "压缩包路径: $zipPath" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
} else {
    Write-Host "打包失败，未生成压缩包。" -ForegroundColor Red
}
