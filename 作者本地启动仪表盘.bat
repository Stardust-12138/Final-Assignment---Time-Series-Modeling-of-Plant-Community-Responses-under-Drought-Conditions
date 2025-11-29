@echo off
echo 正在激活环境并启动...

:: 1. 这里修改为你的 Anaconda 激活脚本的路径 (根据你的安装位置修改)
call "D:\anaconda\Scripts\activate.bat" 

:: 2. 激活你创建的虚拟环境名称 (假设名字叫 myenv)
call conda activate gcn_env 

:: 3. 切换目录并运行
cd /d "%~dp0"
streamlit run app.py

pause