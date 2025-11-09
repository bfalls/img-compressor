# zip.ps1
Add-Type -AssemblyName System.IO.Compression.FileSystem

$projectDir = $args[0]
if (-not $projectDir) { Write-Error "Project directory not provided!"; exit 1 }

$zipPath    = "$($projectDir)img-compressor-src.zip"
Write-Host "Zipping from: $projectDir" -ForegroundColor Cyan
Write-Host "Output zip: $zipPath" -ForegroundColor Green

# Overwrite: delete existing zip
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

# Open new zip
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Create')

Get-ChildItem -Path $projectDir -Recurse -File | Where-Object {
    # Exclude folders by full path (not just ending)
    $_.FullName -notmatch '\\(vcpkg_installed|Debug|Release|bin|obj|x64|\.vs|packages|node_modules)(\\|$)' -and
    # Exclude common build/output extensions
    $_.Extension -notmatch '^\.(jpg|jpeg|exp|lib|zip|exe|dll|pdb|suo|user|cache|log)$'
} | ForEach-Object {
    $entryName = $_.FullName.Substring($projectDir.Length).TrimStart('\')
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $zip, $_.FullName, $entryName, 'Optimal'
    )
}

$zip.Dispose()
Write-Host "Zipped to $($zipPath)" -ForegroundColor Green
