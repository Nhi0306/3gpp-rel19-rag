# Chia se web local qua Cloudflare Tunnel (link trycloudflare.com).
# Chay:  powershell -ExecutionPolicy Bypass -File scripts\share_web.ps1
# Giu may bat + cua so nay mo de link con hoat dong.
param(
    [int]$Port = 8010,
    [string]$Profile = "cheap",
    [string]$SynthModel = "google/gemini-2.5-flash",
    [int]$DailyLimit = 100,
    [int]$RatePerMin = 10,
    [int]$MaxChars = 2000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path tools, logs | Out-Null

$cf = Join-Path $root "tools\cloudflared.exe"
if (-not (Test-Path $cf)) {
    Write-Host "Dang tai cloudflared (~53 MB)..."
    curl.exe -L --fail --max-time 300 -o $cf "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
}

$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    $env:RAG_PROFILE = $Profile
    $env:SYNTH_MODEL = $SynthModel
    $env:RAG_MAX_REFINE = "1"
    $env:RAG_DAILY_LIMIT = "$DailyLimit"
    $env:RAG_RATE_PER_MIN = "$RatePerMin"
    $env:RAG_MAX_QUESTION_CHARS = "$MaxChars"
    Write-Host "Khoi dong server o cong $Port ..."
    Start-Process -FilePath python `
        -ArgumentList "scripts\rag_serve.py", "--host", "127.0.0.1", "--port", "$Port" `
        -RedirectStandardOutput "logs\share-serve.out" `
        -RedirectStandardError "logs\share-serve.err" `
        -WindowStyle Hidden
    for ($i = 0; $i -lt 24; $i++) {
        Start-Sleep -Seconds 5
        try {
            Invoke-WebRequest "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 10 | Out-Null
            break
        } catch { }
    }
}

Remove-Item logs\share-tunnel.out, logs\share-tunnel.err -ErrorAction SilentlyContinue
Start-Process -FilePath $cf `
    -ArgumentList "tunnel", "--url", "http://127.0.0.1:$Port", "--no-autoupdate" `
    -RedirectStandardOutput "logs\share-tunnel.out" `
    -RedirectStandardError "logs\share-tunnel.err" `
    -WindowStyle Hidden

$url = $null
for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 3
    $txt = (Get-Content logs\share-tunnel.err -Raw -ErrorAction SilentlyContinue) +
           (Get-Content logs\share-tunnel.out -Raw -ErrorAction SilentlyContinue)
    $m = [regex]::Match($txt, "https://[a-z0-9-]+\.trycloudflare\.com")
    if ($m.Success) { $url = $m.Value; break }
}

if ($url) {
    Write-Host ""
    Write-Host "Link cong khai: $url" -ForegroundColor Green
    Write-Host "May phai bat; tat tunnel/server thi link chet. Chay lai script de lay link moi."
} else {
    Write-Host "Khong lay duoc URL. Xem logs\share-tunnel.err" -ForegroundColor Red
}
