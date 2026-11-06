# Cai dat Task Scheduler: moi tuan kiem tra 3GPP va cap nhat Qdrant.
# Chay: powershell -ExecutionPolicy Bypass -File scripts\install_weekly_task.ps1
param(
    [string]$TaskName = '3GPP-Rel19-WeeklyUpdate',
    [string]$Day = 'Sunday',
    [string]$Time = '03:00'
)

$root = 'D:\viettel\3GPP'
$log = Join-Path $root 'logs\update-weekly.log'
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

$action = New-ScheduledTaskAction `
    -Execute 'cmd.exe' `
    -Argument "/c python -X utf8 scripts\3gpp_rel19_pipeline.py update >> `"$log`" 2>&1"` `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -RunOnlyIfNetworkAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description '3GPP Rel-19: kiem tra du lieu moi va cap nhat Qdrant hang tuan' `
    -Force | Out-Null

Write-Host "Da cai task '$TaskName' - chay $Day hang tuan luc $Time"
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
