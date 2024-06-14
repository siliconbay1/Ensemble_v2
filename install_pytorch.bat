set venv=ensemble

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

call conda deactivate
call conda env remove -n %venv%
call conda create -n %venv% python=3.9
call conda activate %venv%
:: call conda install pytorch torchvision==0.13.0 torchaudio cudatoolkit=11.3 -c pytorch
call conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
call conda install notebook ipykernel
call python -m ipykernel install --user --name pytorch --display-name pytorch
call pip install -r _requirements.txt

cmd /k