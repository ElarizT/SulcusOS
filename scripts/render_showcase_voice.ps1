param(
    [string]$OutputDirectory = "showcase\voice"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$segments = @(
    @{ File = "01_intro.wav"; Text = "Meet Sulcus OS: an experimental multi-agent operating system." },
    @{ File = "02_runtime.wav"; Text = "Agents are easy to launch. Teams need lifecycles, communication, memory, isolation, and recovery, visible in one runtime." },
    @{ File = "03_dashboard.wav"; Text = "The live dashboard is the control plane. Start a workflow, then watch its agent tree, processes, mailbox traffic, memory pages, and isolation update in real time." },
    @{ File = "04_research.wav"; Text = "A planner fans work out to three specialists. Typed I P C messages converge at a synthesizer, then a critic scores the result. Six agents. One traceable workflow." },
    @{ File = "05_recovery.wav"; Text = "This crash probe fails on purpose. The supervisor detects it, starts P I D one oh five, and verifies the replacement, preserving the full trace." },
    @{ File = "06_memory.wav"; Text = "Run agents in process, or as isolated children. Persist memory, replay structured timelines, and control tool permissions." },
    @{ File = "07_close.wav"; Text = "Sulcus OS. Build agent systems you can see, understand, and trust." }
)

$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice("Microsoft David Desktop")
$voice.Rate = 0
$voice.Volume = 100

foreach ($segment in $segments) {
    $path = Join-Path $OutputDirectory $segment.File
    $voice.SetOutputToWaveFile($path)
    $voice.Speak($segment.Text)
    $voice.SetOutputToNull()
}

$voice.Dispose()
Write-Output "Rendered $($segments.Count) narration segments to $OutputDirectory"
