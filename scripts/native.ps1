param(
    [ValidateSet('build-tests', 'test', 'build-benchmarks', 'benchmark')]
    [string]$Action = 'test',
    [string]$Filter = '',
    [ValidateRange(0, 1000000)]
    [int]$Warmup = 10,
    [ValidateRange(1, 1000000)]
    [int]$Samples = 51
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Visual Studio Installer vswhere.exe was not found.'
}
$visualStudio = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) {
    throw 'A Visual Studio installation with the C++ toolchain was not found.'
}
$vcvars = Join-Path $visualStudio 'VC\Auxiliary\Build\vcvars64.bat'
$cmake = Join-Path $visualStudio `
    'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$ctest = Join-Path $visualStudio `
    'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe'
$ninja = Join-Path $visualStudio `
    'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
foreach ($required in @($vcvars, $cmake, $ctest, $ninja)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required native build tool was not found: $required"
    }
}
if (-not $env:CUDA_PATH) {
    throw 'CUDA_PATH is not set.'
}
$nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc)) {
    throw "nvcc was not found: $nvcc"
}

# Import the compiler environment without coupling the repository to one MSVC
# toolset version. Values may contain '=' (notably command prompt metadata), so
# split only on the first delimiter.
$environmentLines = & cmd.exe /d /s /c "`"$vcvars`" >nul && set"
foreach ($line in $environmentLines) {
    $delimiter = $line.IndexOf('=')
    if ($delimiter -gt 0) {
        $name = $line.Substring(0, $delimiter)
        $value = $line.Substring($delimiter + 1)
        Set-Item -LiteralPath "Env:$name" -Value $value
    }
}

$buildDirectory = Join-Path $repository 'build\native'
$torchCMake = Join-Path $repository `
    '.venv\Lib\site-packages\torch\share\cmake'
if (-not (Test-Path -LiteralPath $torchCMake)) {
    throw "The verified repository PyTorch CMake package was not found: $torchCMake"
}
$torchLibrary = Join-Path $repository '.venv\Lib\site-packages\torch\lib'
$env:PATH = "$torchLibrary;$env:PATH"
$withBenchmarks = $Action -in @('build-benchmarks', 'benchmark')
$benchmarkValue = if ($withBenchmarks) { 'ON' } else { 'OFF' }
& $cmake -S $repository -B $buildDirectory -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_CUDA_COMPILER=$nvcc" `
    -DCMAKE_CUDA_ARCHITECTURES=120 `
    "-DCMAKE_PREFIX_PATH=$torchCMake" `
    -DCMAKE_BUILD_TYPE=Release `
    -DFLUX_BUILD_NATIVE_TESTS=ON `
    "-DFLUX_BUILD_NATIVE_BENCHMARKS=$benchmarkValue"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmake --build $buildDirectory --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq 'test') {
    & $ctest --test-dir $buildDirectory --output-on-failure -C Release
    exit $LASTEXITCODE
}
if ($Action -eq 'benchmark') {
    $executable = Join-Path $buildDirectory `
        'csrc\benchmarks\flux_cuda_microbenchmarks.exe'
    $arguments = @('--warmup', $Warmup, '--samples', $Samples)
    if ($Filter) {
        $arguments += @('--filter', $Filter)
    }
    & $executable @arguments
    exit $LASTEXITCODE
}
