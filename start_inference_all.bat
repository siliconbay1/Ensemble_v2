set venv=pytorch
set exp_name=ensemble_x4-ImageNet_GAN

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

:: Change directory to the relative path that's needed for script
:: cd Download

:: Run script at this location
:: call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\main.py"
:: PAUSE

:: call python inference.py --inputs_path ./figure/comic_lr.png --model_weights_path ./results/%exp_name%/g_best.pth.tar --output_path ./figure/comic_sr_epoches1000.png

:: call FOR /L %i IN (1,1,5) DO ECHO %i

call python inference.py --inputs_path ./figure/comic.png --model_weights_path ./results/%exp_name%/g_best.pth.tar --output_path ./figure/comic_sr_epoches_best.png

@echo off
@REM for /L %%n in (0,1,47) do (echo.> %%n.txt)
for /L %%n in (59,1,130) do ( 
    call python inference.py --inputs_path ./figure/comic.png --model_weights_path ./samples/%exp_name%/g_epoch_%%n.pth.tar --output_path ./figure/comic_sr_epoches%%n.png
    )

cmd /k