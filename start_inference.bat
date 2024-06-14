set venv=ensemble

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

:: Change directory to the relative path that's needed for script
:: cd Download

:: Run script at this location
:: call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\main.py"
:: PAUSE

call python espcn+fsrcnn_inference.py --inputs_path ./figure/103070_LR.png --model_weights_path ./results/+espcn+fsrcnn+w1w2+2channel+set14/epoch_60_testset3_best.pth.tar --output_path sr_espcn+fsrcnn

call python res_espcn+fsrcnn_inference.py --inputs_path ./figure/103070_LR.png --model_weights_path ./results/+espcn+fsrcnn+2channel+weighted_residual/epoch_24_testset2_best.pth.tar --output_path res_espcn+fsrcnn

call python fsrcnn+vdsr_inference.py --inputs_path ./figure/103070_LR.png --model_weights_path ./results/+fsrcnn+vdsr+w1w2+2channel+extendedset/epoch_89_testset0_best.pth.tar --output_path sr_fsrcnn+vdsr

call python res_fsrcnn+vdsr_inference.py --inputs_path ./figure/103070_LR.png --model_weights_path ./results/+fsrcnn+vdsr+2channel+weighted_residual/epoch_47_testset0_best.pth.tar --output_path res_fsrcnn+vdsr

:: call FOR /L %i IN (1,1,5) DO ECHO %i

cmd /k