set venv=ensemble

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

:: Change directory to the relative path that's needed for script
:: cd Download

:: Run script at this location
:: call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\main.py"
:: PAUSE

:: call python train_ensemble.py > ./log/result_srgan.txt

:: rmdir /s /q samples

call python train_ensemble.py

:: call python train.py > ./log/train_result_20230430.txt

cmd /k