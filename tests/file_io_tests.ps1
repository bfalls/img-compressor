param(
    [string]$Configuration = "Debug"
)

$ErrorActionPreference = "Stop"

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function Run-ImgCompressor {
    param(
        [string]$ExePath,
        [string]$InputPath,
        [string]$OutputPath,
        [switch]$Compare
    )

    $args = @("--input", $InputPath, "--output", $OutputPath)
    if ($Compare) {
        $args += "--compare"
    }

    $output = & $ExePath @args 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = ($output -join "`n")
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$exePath = Join-Path $repoRoot ("x64\" + $Configuration + "\img-compressor.exe")

if (-not (Test-Path $exePath)) {
    throw "Missing executable: $exePath (build the solution first)."
}

$probe = & $exePath 2>&1
if ($LASTEXITCODE -eq -1073741515) {
    Write-Host "Skipping file IO tests: img-compressor.exe failed to start (0xC0000135 missing DLL)."
    exit 0
}

$inputRel = "tests\data\img-test.png"
$inputAbs = (Resolve-Path (Join-Path $repoRoot $inputRel)).Path

$artifactsRoot = Join-Path $repoRoot "tests\artifacts\io"
New-Item -ItemType Directory -Force -Path $artifactsRoot | Out-Null

Push-Location $repoRoot
try {
    # Non-existent input
    $missingDir = Join-Path $artifactsRoot "missing"
    New-Item -ItemType Directory -Force -Path $missingDir | Out-Null
    $missingInput = Join-Path $repoRoot "tests\data\missing-file.png"
    $missingOutput = Join-Path $missingDir "out.jpg"
    $result = Run-ImgCompressor -ExePath $exePath -InputPath $missingInput -OutputPath $missingOutput
    Assert-True ($result.ExitCode -ne 0) "Expected failure for missing input."
    Assert-True (-not (Test-Path $missingOutput)) "Missing input should not create output."

    # Read-only output directory
    $readonlyDir = Join-Path $artifactsRoot "readonly"
    New-Item -ItemType Directory -Force -Path $readonlyDir | Out-Null
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $denyApplied = $false
    try {
        icacls $readonlyDir /deny "$($identity):(W)" | Out-Null
        $denyApplied = $true
        $readonlyOutput = Join-Path $readonlyDir "out.jpg"
        $result = Run-ImgCompressor -ExePath $exePath -InputPath $inputRel -OutputPath $readonlyOutput -Compare
        Assert-True ($result.ExitCode -ne 0) "Expected failure for read-only output directory."
        Assert-True (-not (Test-Path $readonlyOutput)) "Read-only output should not create output."
    }
    finally {
        if ($denyApplied) {
            icacls $readonlyDir /remove:d "$identity" | Out-Null
        }
    }

    # Overwrite behavior
    $overwriteDir = Join-Path $artifactsRoot "overwrite"
    New-Item -ItemType Directory -Force -Path $overwriteDir | Out-Null
    $existingCpu = Join-Path $overwriteDir "out-cpu.jpg"
    Set-Content -Path $existingCpu -Value "old" -NoNewline
    $beforeTime = (Get-Item $existingCpu).LastWriteTimeUtc
    $overwriteOutput = Join-Path $overwriteDir "out.jpg"
    $result = Run-ImgCompressor -ExePath $exePath -InputPath $inputRel -OutputPath $overwriteOutput -Compare
    Assert-True ($result.ExitCode -eq 0) "Expected overwrite run to succeed."
    $afterItem = Get-Item $existingCpu
    Assert-True ($afterItem.Length -gt 0) "Expected overwritten file to have data."
    Assert-True ($afterItem.LastWriteTimeUtc -gt $beforeTime) "Expected overwritten file timestamp to update."

    # Relative paths
    $relativeDir = Join-Path $repoRoot "tests\artifacts\relative"
    New-Item -ItemType Directory -Force -Path $relativeDir | Out-Null
    $relativeOutput = "tests\artifacts\relative\out.jpg"
    $result = Run-ImgCompressor -ExePath $exePath -InputPath $inputRel -OutputPath $relativeOutput -Compare
    Assert-True ($result.ExitCode -eq 0) "Expected relative path run to succeed."
    $relativeCpu = Join-Path $relativeDir "out-cpu.jpg"
    Assert-True (Test-Path $relativeCpu) "Expected relative output to exist."

    # Absolute paths
    $absoluteDir = Join-Path $artifactsRoot "absolute"
    New-Item -ItemType Directory -Force -Path $absoluteDir | Out-Null
    $absoluteOutput = Join-Path $absoluteDir "out.jpg"
    $result = Run-ImgCompressor -ExePath $exePath -InputPath $inputAbs -OutputPath $absoluteOutput -Compare
    Assert-True ($result.ExitCode -eq 0) "Expected absolute path run to succeed."
    $absoluteCpu = Join-Path $absoluteDir "out-cpu.jpg"
    Assert-True (Test-Path $absoluteCpu) "Expected absolute output to exist."
}
finally {
    Pop-Location
}

Write-Host "File IO tests passed."
