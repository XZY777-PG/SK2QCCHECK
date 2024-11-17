@echo off
REM 设置 Python 可执行文件的路径
set PYTHON_PATH=C:\Users\dtcskiiprint04.im\AppData\Local\miniforge3\envs\yolo\python.exe

REM 设置你的 Python 脚本的路径
set SCRIPT_PATH=C:\develop\PaddleOCR-main\PaddleOCR-main\pagetest 3 7.py

REM 运行 Python 脚本
"%PYTHON_PATH%" "%SCRIPT_PATH%"

REM 暂停，以便查看输出
pause
