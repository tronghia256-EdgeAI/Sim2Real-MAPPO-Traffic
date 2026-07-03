# eval_tricks_local.ps1 — re-evaluate MAPPO vs baselines with the two
# eval-time tricks (no retraining needed). Run from the project root:
#
#   powershell -ExecutionPolicy Bypass -File scripts\final2\eval_tricks_local.ps1
#   # or a subset:
#   ... -Only trick1            # OOD high demand only
#   ... -Only trick2            # frequency boost only
#   ... -Only combo             # both tricks together
#
# TRICK 1 — OOD High Demand (VI-D scenario, uniform 1.67 veh/s x network scale):
#   heavy saturation is where adaptive control should separate from gap-based
#   actuated (under saturation actuated rarely gaps out and degrades toward
#   max-green fixed-time). Uses sumo_config_ood_high.sumocfg (already generated).
#
# TRICK 2 — Eval-time Frequency Boost (--learned-step-length 3):
#   MAPPO decides every 3 s instead of the 5 s training cadence; baselines keep
#   5 s; the 5400 s horizon and the 15 s min-green are unchanged. 3 s is the
#   hard floor (yellow_time=3 must fit inside one decision step). This narrows
#   actuated's per-second reaction advantage without retraining.
#
# PAPER HONESTY: both are eval-protocol changes. Report trick 1 as the OOD
# robustness result (sec VI-D) and trick 2 as a decision-frequency sensitivity
# analysis with the boosted interval stated explicitly. Do NOT silently swap
# either into the main Table I protocol.
#
# Each run writes results/paper1_mappo/eval_tables/n3_grid_<ts>/{summary.json,
# table.csv}; the trick -> output-dir mapping is appended to
# results/paper1_mappo/tricks_evallogs/MANIFEST.txt (summary.json also records
# learned_step_length so runs stay distinguishable).

param(
    [string]$Only = "all",   # all | trick1 | trick2 | combo
    [string[]]$Seeds = @("42", "123", "456", "789", "1337"),
    [int]$LearnedStep = 3,
    [string]$Ckpt = "results/paper1_mappo/ckpts_clean/old_main/n3_grid_mappo_proxy_seed*/best_model.pt"
)

$env:PYTHONIOENCODING = "utf-8"
$Methods = @("mappo", "webster", "actuated", "maxpressure", "sotl", "fixed")
$OodCfg = "sumo_configs/networks/n3_grid/sumo_config_ood_high.sumocfg"
$LaneGroups = "sumo_configs/networks/n3_grid/lane_groups.json"
$LogDir = "results/paper1_mappo/tricks_evallogs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Manifest = Join-Path $LogDir "MANIFEST.txt"

function Get-NewestTable {
    $d = Get-ChildItem "results/paper1_mappo/eval_tables" -Directory |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($null -ne $d) { return $d.FullName } else { return "<none>" }
}

function Run-Eval([string]$Tag, [string[]]$ExtraArgs) {
    Write-Host "`n===== $Tag =====" -ForegroundColor Cyan
    $argv = @("experiment/runners/eval_compare.py",
              "--checkpoint", $Ckpt,
              "--methods") + $Methods + @("--seeds") + $Seeds + $ExtraArgs
    python @argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Tag] eval_compare FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        return
    }
    $out = Get-NewestTable
    Add-Content $Manifest "$(Get-Date -Format s)  $Tag -> $out"
    Write-Host "[$Tag] table: $out" -ForegroundColor Green
}

if (($Only -eq "all") -or ($Only -eq "trick1")) {
    Run-Eval "TRICK1_OOD_HIGH" @(
        "--sumo-cfg", $OodCfg, "--lane-groups", $LaneGroups)
}
if (($Only -eq "all") -or ($Only -eq "trick2")) {
    Run-Eval "TRICK2_FREQBOOST_${LearnedStep}s" @(
        "--network", "n3_grid",
        "--learned-step-length", "$LearnedStep")
}
if (($Only -eq "all") -or ($Only -eq "combo")) {
    Run-Eval "COMBO_OOD_HIGH+FREQBOOST_${LearnedStep}s" @(
        "--sumo-cfg", $OodCfg, "--lane-groups", $LaneGroups,
        "--learned-step-length", "$LearnedStep")
}

Write-Host "`nDone. Mapping in $Manifest"
Get-Content $Manifest -ErrorAction SilentlyContinue | Select-Object -Last 5
