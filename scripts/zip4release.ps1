<#
.SYNOPSIS
  Packages a Windows release ZIP for GitHub Releases and writes SHA256SUMS.txt.

.EXAMPLE
  .\scripts\zip4release.ps1 -Version v0.1.0 -ExePath .\x64\Release\img-compressor.exe
#>

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidatePattern('^v?\d+\.\d+\.\d+(-[0-9A-Za-z\.\-]+)?$')]
  [string]$Version,

  [Parameter(Mandatory = $true)]
  [ValidateNotNullOrEmpty()]
  [string]$ExePath,

  [Parameter()]
  [ValidateNotNullOrEmpty()]
  [string]$Name = 'img-compressor',

  [Parameter()]
  [ValidateNotNullOrEmpty()]
  [string]$Platform = 'win-x64',

  [Parameter()]
  [ValidateNotNullOrEmpty()]
  [string]$OutDir = '.\dist',

  [Parameter()]
  [string[]]$IncludeFiles = @()
)

$ErrorActionPreference = 'Stop'

# Anchor everything to repo root (parent of scripts/)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Normalize-Version([string]$v) {
  if ($v.StartsWith('v')) { return $v }
  return "v$v"
}

$ver = Normalize-Version $Version

# Resolve EXE path (relative paths should be relative to repo root)
$exeAbs = $ExePath
if (-not [System.IO.Path]::IsPathRooted($exeAbs)) {
  $exeAbs = Join-Path $repoRoot $ExePath
}
if (-not (Test-Path -LiteralPath $exeAbs)) {
  throw "ExePath not found: $exeAbs"
}
if ([System.IO.Path]::GetExtension($exeAbs).ToLowerInvariant() -ne '.exe') {
  throw "ExePath must point to an .exe: $exeAbs"
}
$exeAbs = (Resolve-Path -LiteralPath $exeAbs).Path

# Output paths
$outAbs = $OutDir
if (-not [System.IO.Path]::IsPathRooted($outAbs)) {
  $outAbs = Join-Path $repoRoot $OutDir
}
New-Item -ItemType Directory -Force -Path $outAbs | Out-Null

$zipName = "$Name-$ver-$Platform.zip"
$zipPath = Join-Path $outAbs $zipName
$shaPath = Join-Path $outAbs 'SHA256SUMS.txt'

Write-Host "RepoRoot:   $repoRoot"
Write-Host "Version:    $ver"
Write-Host "Exe:        $exeAbs"
Write-Host "OutDir:     $outAbs"
Write-Host "Zip:        $zipPath"

# Stage into a temp folder so ZIP has files at the root
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("zip4release_" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

try {
  Copy-Item -LiteralPath $exeAbs -Destination (Join-Path $tempRoot (Split-Path $exeAbs -Leaf)) -Force

  $readme = Join-Path $repoRoot 'README.md'
  if (Test-Path -LiteralPath $readme) {
    Copy-Item -LiteralPath $readme -Destination (Join-Path $tempRoot 'README.md') -Force
  }

  $license = Join-Path $repoRoot 'LICENSE'
  if (Test-Path -LiteralPath $license) {
    Copy-Item -LiteralPath $license -Destination (Join-Path $tempRoot 'LICENSE') -Force
  }

  foreach ($p in $IncludeFiles) {
    $abs = $p
    if (-not [System.IO.Path]::IsPathRooted($abs)) {
      $abs = Join-Path $repoRoot $p
    }
    if (Test-Path -LiteralPath $abs) {
      $leaf = Split-Path $abs -Leaf
      Copy-Item -LiteralPath $abs -Destination (Join-Path $tempRoot $leaf) -Force
    } else {
      Write-Warning "IncludeFiles not found, skipping: $p"
    }
  }

  # Create ZIP from staged contents
  if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
  Compress-Archive -Path (Join-Path $tempRoot '*') -DestinationPath $zipPath -Force

  # Write checksums for all files in dist (at least the zip)
  if (Test-Path -LiteralPath $shaPath) { Remove-Item -LiteralPath $shaPath -Force }

  $distFiles = Get-ChildItem -LiteralPath $outAbs -File | Sort-Object Name
  foreach ($f in $distFiles) {
    $hash = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($f.Name)" | Out-File -LiteralPath $shaPath -Append -Encoding ascii
  }

  Write-Host "SHA256:     $shaPath"
  Write-Host "Done."
}
finally {
  if (Test-Path -LiteralPath $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
  }
}
