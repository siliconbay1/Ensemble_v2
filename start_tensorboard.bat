::taskkill /IM chrome.exe
start chrome.exe http://localhost:6006/?darkMode=true#timeseries

set venv=ensemble

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

:: Change directory to the relative path that's needed for script
:: cd Download

:: Run script at this location
:: call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\main.py"
:: PAUSE

tensorboard --logdir ./samples/logs
:: --port=8083

cmd /k