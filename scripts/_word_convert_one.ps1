param([string]$In, [string]$Out)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
$w.AutomationSecurity = 3
$d = $w.Documents.Open($In, $false, $true, $false)
$d.SaveAs2($Out, 16)
$d.Close(0)
$w.Quit()
