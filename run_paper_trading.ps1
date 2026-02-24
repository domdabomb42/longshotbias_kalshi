param(
    [string]$Python = "python",
    [double]$InitialCash = 1000.0,
    [double]$Stake = 10.0,
    [double]$MinEv = 0.0,
    [double]$PollSeconds = 60.0,
    [int]$MaxMarkets = 2000,
    [int]$MaxOpenPositions = 200,
    [int]$MaxNewOrdersPerLoop = 25,
    [ValidateSet("taker", "maker", "both")]
    [string]$Liquidity = "both",
    [string]$LiveStatus = "open",
    [switch]$AllowIlliquid
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$outputDir = Join-Path $root "outputs\paper_trading"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $outputDir "paper_trader_$ts.log"

$argsList = @(
    "-m", "kalshi_longshot_bias.cli", "paper-trade",
    "--output-dir", $outputDir,
    "--initial-cash", "$InitialCash",
    "--stake", "$Stake",
    "--min-ev", "$MinEv",
    "--poll-seconds", "$PollSeconds",
    "--max-markets", "$MaxMarkets",
    "--max-open-positions", "$MaxOpenPositions",
    "--max-new-orders-per-loop", "$MaxNewOrdersPerLoop",
    "--liquidity", $Liquidity,
    "--live-status", $LiveStatus
)

if ($AllowIlliquid) {
    $argsList += "--allow-illiquid"
}

Write-Host "Starting paper trader..."
Write-Host "Output: $outputDir"
Write-Host "Log: $logPath"

& $Python @argsList 2>&1 | Tee-Object -FilePath $logPath
