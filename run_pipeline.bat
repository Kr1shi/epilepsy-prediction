@echo off
set LOG_FILE=pipeline_log.txt

echo Starting pipeline at %DATE% %TIME% > %LOG_FILE%
echo ---------------------------------------- >> %LOG_FILE%

echo Cleaning up previous models and preprocessed data...
echo Cleaning up previous models and preprocessed data... >> %LOG_FILE% 2>&1
if exist model (
    rmdir /s /q model
)
if exist preprocessing (
    rmdir /s /q preprocessing
)

echo Running data segmentation...
echo Running data segmentation... >> %LOG_FILE% 2>&1
python data_segmentation.py >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 goto :error

echo Running data preprocessing...
echo Running data preprocessing... >> %LOG_FILE% 2>&1
python data_preprocessing.py >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 goto :error

echo Running model training...
echo Running model training... >> %LOG_FILE% 2>&1
python train.py >> %LOG_FILE% 2>&1
if %ERRORLEVEL% NEQ 0 goto :error

echo ---------------------------------------- >> %LOG_FILE%
echo Pipeline finished successfully at %DATE% %TIME% >> %LOG_FILE%
echo Pipeline finished successfully. Check %LOG_FILE% for details.
goto :eof

:error
echo ---------------------------------------- >> %LOG_FILE%
echo Pipeline failed at %DATE% %TIME% >> %LOG_FILE%
echo Pipeline failed! Check %LOG_FILE% for details.
exit /b %ERRORLEVEL%
