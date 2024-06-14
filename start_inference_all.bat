set venv=pytorch

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

:: Change directory to the relative path that's needed for script
:: cd Download

:: Run script at this location
:: call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\main.py"
:: PAUSE

:: call python inference.py --inputs_path ./figure/baboon.png --model_weights_path ./results_epoches1000/SRGAN_x4-DIV2K/g_best.pth.tar --output_path ./figure/comic_sr_epoches1000.png > ./log/result_inference.txt

:: call FOR /L %i IN (1,1,5) DO ECHO %i

@echo off
@REM for /L %%n in (1,1,47) do (echo.> %%n.txt)
for /L %%n in (1,1,5) do ( 
    call python inference.py --inputs ./figure/GOPR0372_07_00_000047.png --model_weights_path ./samples/SRGAN_x4-ImageNet/epoch_%%n.pth.tar --output ./figure/minounet_epoches%%n.png
    )

cmd /k