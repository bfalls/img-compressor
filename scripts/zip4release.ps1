<#
.SYNOPSIS
Packages a built Windows release executable into a GitHub-ready ZIP and generates SHA-256 checksums.

.DESCRIPTION
Creates a ZIP containing the provided executable at the archive root, plus optional accompanying files
(defaults to README.md and LICENSE when present). The script writes artifacts to a dist/ directory and
generates a SHA256SUMS.txt that includes every file in the output directory.

.PARAMETER Version
Version string (e.g., v0.1.0 or 0.1.0). Non-prefixed versions are normalized to start with "v" for filenames.

.PARAMETER ExePath
Path to the built executable (.exe). Must exist.

.PARAMETER Name
Base product name used in filenames. Defaults to "img-compressor".

.PARAMETER Platform
Platform identifier used in filenames. Defaults to "win-x64".

.PARAMETER OutDir
Output directory for artifacts. Defaults to ".\dist".

.PARAMETER IncludeFiles
Additional files to include in the ZIP when they exist. Defaults to including README.md and LICENSE when present.

.EXAMPLE
.\zip4release.ps1 -Version v0.1.0 -ExePath .\build\Release\img-compressor.exe

.EXAMPLE
.\zip4release.ps1 -Version 0.2.0-beta.1 -ExePath .\build\Release\img-compressor.exe -IncludeFiles @('CHANGELOG.md')
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$ExePath,

    [string]$Name = 'img-compressor',
    [string]$Platform = 'win-x64',
    [string]$OutDir = '.\dist',
    [string[]]$IncludeFiles
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$repoRoot = $scriptRoot
$versionPattern = '^v?\d+\.\d+\.\d+(-[0-9A-Za-z\.\-]+)?$'

function Resolve-WithRepoRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path -LiteralPath $Path).ProviderPath
    }

    $combined = Join-Path -Path $repoRoot -ChildPath $Path
    return (Resolve-Path -LiteralPath $combined).ProviderPath
}

function Add-OptionalFile {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.List[string]]$TargetList,

        [Parameter(Mandatory = $true)]
        [string]$CandidatePath,

        [Parameter()]
        [switch]$WarnIfMissing
    )

    if (-not (Test-Path -LiteralPath $CandidatePath)) {
        if ($WarnIfMissing) {
            Write-Warning "Skipping missing file: $CandidatePath"
        }
        return
    }

    $TargetList.Add((Resolve-Path -LiteralPath $CandidatePath).ProviderPath) | Out-Null
}

if ($Version -notmatch $versionPattern) {
    throw "Version '$Version' is invalid. Expected format: v0.1.0 or 0.1.0 (pre-release like v0.1.0-beta.1 allowed)."
}

$versionNormalized = if ($Version.StartsWith('v')) { $Version } else { "v$Version" }

$outDirFullPath = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    [System.IO.Path]::GetFullPath($OutDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path -Path $repoRoot -ChildPath $OutDir))
}

if (-not (Test-Path -LiteralPath $outDirFullPath)) {
    Write-Host "Creating output directory: $outDirFullPath"
    New-Item -ItemType Directory -Path $outDirFullPath -Force | Out-Null
}

if (-not $PSBoundParameters.ContainsKey('IncludeFiles') -or -not $IncludeFiles) {
    $IncludeFiles = @('README.md', 'LICENSE')
}

$exeFullPath = Resolve-WithRepoRoot -Path $ExePath
if (-not (Test-Path -LiteralPath $exeFullPath)) {
    throw "Executable not found at path: $ExePath"
}

if ([System.IO.Path]::GetExtension($exeFullPath) -ne '.exe') {
    throw "Executable path must end with '.exe': $ExePath"
}

$filesToPackage = New-Object System.Collections.Generic.List[string]
Add-OptionalFile -TargetList $filesToPackage -CandidatePath $exeFullPath -WarnIfMissing

foreach ($file in $IncludeFiles) {
    try {
        $resolved = Resolve-WithRepoRoot -Path $file
    } catch {
        $resolved = $null
    }

    if ($null -ne $resolved) {
        Add-OptionalFile -TargetList $filesToPackage -CandidatePath $resolved
    } else {
        Write-Warning "Skipping missing file: $file"
    }
}

if ($filesToPackage.Count -eq 0) {
    throw "No files found to package."
}

$zipName = "$Name-$versionNormalized-$Platform.zip"
$zipPath = Join-Path -Path $outDirFullPath -ChildPath $zipName
$checksumPath = Join-Path -Path $outDirFullPath -ChildPath 'SHA256SUMS.txt'

Write-Host "Normalized version: $versionNormalized"
Write-Host "Resolved executable: $exeFullPath"
Write-Host "Output ZIP: $zipPath"
Write-Host "Checksum file: $checksumPath"

$stagingPath = Join-Path -Path $outDirFullPath -ChildPath ("zip-stage-{0}" -f ([guid]::NewGuid().ToString('N')))
New-Item -ItemType Directory -Path $stagingPath -Force | Out-Null

try {
    foreach ($file in $filesToPackage) {
        $destination = Join-Path -Path $stagingPath -ChildPath (Split-Path -Path $file -Leaf)
        if (Test-Path -LiteralPath $destination) {
            throw "File name conflict while staging: $destination already exists."
        }

        Copy-Item -LiteralPath $file -Destination $destination -Force
    }

    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    $stagedFiles = Get-ChildItem -Path $stagingPath -File
    Compress-Archive -LiteralPath $stagedFiles.FullName -DestinationPath $zipPath -Force
}
finally {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force -ErrorAction SilentlyContinue
}

$artifacts = Get-ChildItem -Path $outDirFullPath -File | Sort-Object -Property Name
$checksumLines = @()
foreach ($artifact in $artifacts) {
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $artifact.FullName
    $checksumLines += ("{0}  {1}" -f $hash.Hash, $artifact.Name)
}

Set-Content -Path $checksumPath -Value $checksumLines -Encoding ASCII
Write-Host "Packaging complete."
