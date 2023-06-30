set venv=ensemble

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

:: Change directory to the relative path that's needed for script
:: cd Download

:: Run script at this location
:: call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\main.py"
:: PAUSE

call python inference.py --inputs_path ./figure/comic.png --model_weights_path ./samples/ensemble_x4-T91/g_epoch_10.pth.tar --output_path ./figure/comic_sr_epoch_best.png

cmd /k