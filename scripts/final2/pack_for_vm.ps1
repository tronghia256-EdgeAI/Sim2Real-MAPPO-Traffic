# pack_for_vm.ps1 - run LOCALLY (Windows, from project root) BEFORE creating the VM.
# Builds vm2_payload.tar.gz containing everything git does NOT track but the VM needs:
#   1. Clean OLD checkpoints (final_main_05M: mappo/ippo/privileged x 5 seeds)
#      -> required for the merged n=10 eval on the VM.
#   2. Demand route files for n2_corridor (REGENERATED, 0.80 veh/s capped) and
#      n3_grid (byte-identical to the files the old campaign trained/evaled on
#      - n3 eval demand must stay exactly 14888 vehicles).
#
# Usage:  powershell -File scripts\final2\pack_for_vm.ps1
# Then:   scp vm2_payload.tar.gz <user>@<vm>:~/Sim2Real-MAPPO-Traffic/
#         (extract at the repo root on the VM: tar -xzf vm2_payload.tar.gz)

$ErrorActionPreference = "Stop"
if (-not (Test-Path "scripts/final2/collect_checkpoints.py")) {
    Write-Error "Run from the project root."
}

# -- 1. collect clean old checkpoints ---------------------------------------
python scripts/final2/collect_checkpoints.py `
    --models-dir results/paper1_mappo/final_main_05M/models `
    --out        results/paper1_mappo/ckpts_clean/old_main
if ($LASTEXITCODE -ne 0) { Write-Error "collect_checkpoints failed - fix before packing." }

# sanity: expect 15 jobs (5 mappo_proxy + 5 ippo_proxy + 5 mappo_privileged)
$jobs = Get-ChildItem results/paper1_mappo/ckpts_clean/old_main -Directory
Write-Output "old_main jobs collected: $($jobs.Count) (expect 15)"
if ($jobs.Count -ne 15) { Write-Warning "expected 15 jobs, got $($jobs.Count) - check listing above" }

# sanity: n2 demand must be the CAPPED version (~2837 vehicles), not the stale 11272
$n2m = Select-String -Path sumo_configs/networks/n2_corridor/demand_train.rou.xml -Pattern '<vehicle ' -AllMatches
$n2c = ($n2m | ForEach-Object { $_.Matches.Count } | Measure-Object -Sum).Sum
Write-Output "n2 demand_train vehicles: $n2c (expect ~2837; STALE pre-fix file had 11272)"
if ($n2c -gt 5000) { Write-Error "n2 demand looks STALE - rerun: python scripts/generate_demand.py --network n2_corridor" }

# -- 2. build the file list and tar -----------------------------------------
$files = @()
$files += Get-ChildItem sumo_configs/networks/n2_corridor/*.rou.xml
$files += Get-ChildItem sumo_configs/networks/n3_grid/*.rou.xml
$files += Get-ChildItem results/paper1_mappo/ckpts_clean/old_main -Recurse -File

$rel = $files | ForEach-Object {
    ((Resolve-Path -Relative $_.FullName) -replace '\\', '/') -replace '^\./', ''
}
$rel | Set-Content -Encoding ascii vm2_payload_list.txt
tar -czf vm2_payload.tar.gz -T vm2_payload_list.txt
Remove-Item vm2_payload_list.txt

$sz = [math]::Round((Get-Item vm2_payload.tar.gz).Length / 1MB, 1)
Write-Output "`nvm2_payload.tar.gz ready ($sz MB) - scp it to the VM repo root and extract there."
