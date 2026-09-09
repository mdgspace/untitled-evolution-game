param([string]$Code, [string]$Tool = 'Unity_RunCommand', [int]$TimeoutSeconds = 50)
$ErrorActionPreference = 'Stop'
$expectedProject = 'C:\Users\Vrishin\Desktop\Projects\ai_stuff\untitled-evolution-game\adv_creature_2d_testing'
$connection = Get-ChildItem 'C:\Users\Vrishin\.unity\mcp\connections' -Filter 'bridge-*.json' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $connection) { throw 'No Unity MCP connection registration found' }
$entry = Get-Content -LiteralPath $connection.FullName -Raw | ConvertFrom-Json
if ($entry.project_path -ne $expectedProject) { throw "Unexpected Unity project: $($entry.project_path)" }
$pipeName = ($entry.connection_path -split '\\')[-1]
$pipe = [System.IO.Pipes.NamedPipeClientStream]::new('.', $pipeName, [System.IO.Pipes.PipeDirection]::InOut, [System.IO.Pipes.PipeOptions]::Asynchronous)
try {
    $pipe.Connect(3000)
    $reader = [System.IO.StreamReader]::new($pipe)
    $writer = [System.IO.StreamWriter]::new($pipe, [System.Text.UTF8Encoding]::new($false))
    $writer.AutoFlush = $true
    $handshake = $reader.ReadLine() | ConvertFrom-Json
    if ($handshake.type -ne 'handshake') { throw ($handshake | ConvertTo-Json) }
    Start-Sleep -Milliseconds 250
    $parameters = if ($Tool -eq 'Unity_RunCommand') { @{Code=$Code;Title='Continual locomotion validation'} } else { @{maxEntries=20;includeStackTrace=$true;logTypes='Error,Exception'} }
    $writer.WriteLine((@{type=$Tool;params=$parameters} | ConvertTo-Json -Compress -Depth 8))
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $read = $reader.ReadLineAsync()
        if (!$read.Wait([Math]::Max(1,[int]($deadline-[DateTime]::UtcNow).TotalMilliseconds))) { throw 'Unity command timed out' }
        $line = $read.Result
        if (!$line) { throw 'Unity disconnected' }
        $result = $line | ConvertFrom-Json
        if ($result.type -eq 'command_in_progress') { continue }
        $line
        break
    }
} finally { $pipe.Dispose() }
