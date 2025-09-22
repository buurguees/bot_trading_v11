# activa venv y lanza el updater en bucle
$ErrorActionPreference = "Stop"
$repo = "C:\Users\Alex B\Desktop\bot_trading_v9\bot_trading_v11"
$venv = "$repo\.venv\Scripts\Activate.ps1"
$script = "$repo\core\data\realtime_updater.py"
$log = "$repo\logs\realtime_updater.log"

New-Item -Force -ItemType Directory "$repo\logs" | Out-Null
. $venv
python $script >> $log 2>&1
