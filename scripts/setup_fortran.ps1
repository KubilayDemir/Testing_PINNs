# Install a local Windows GNU Fortran compiler without changing the system PATH.
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$compilerDirectory = Join-Path $repoRoot '.tools/compiler'
$compiler = Join-Path $compilerDirectory 'mingw64/bin/gfortran.exe'
if (Test-Path -LiteralPath $compiler) {
    & $compiler --version
    exit $LASTEXITCODE
}
$archive = Join-Path $repoRoot '.tools/compiler.zip'
New-Item -ItemType Directory -Force $compilerDirectory | Out-Null
$url = 'https://github.com/brechtsanders/winlibs_mingw/releases/download/' +
    '16.2.0posix-14.0.0-ucrt-r1/' +
    'winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r1.zip'
if (!(Test-Path -LiteralPath $archive)) {
    Invoke-WebRequest -Uri $url -OutFile $archive
}
$expected = 'c1f52294597c0b73786b2a78eb5d176d89226d2f21875eab75e783a8b1cefcc4'
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash
if ($actual.ToLowerInvariant() -ne $expected) {
    throw 'Compiler download checksum mismatch.'
}
tar.exe -xf $archive -C $compilerDirectory
if ($LASTEXITCODE -ne 0) { throw 'Compiler extraction failed.' }
& $compiler --version
exit $LASTEXITCODE
