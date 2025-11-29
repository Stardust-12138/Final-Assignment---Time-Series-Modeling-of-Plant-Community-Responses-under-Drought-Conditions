@echo off
echo 正在启动大平原植被监测仪表盘...

:: 切换到当前脚本所在的目录
cd /d "%~dp0"

:: 检查是否安装了 streamlit，如果没有则提示（可选）
:: pip install streamlit pandas numpy plotly folium streamlit-folium torch scikit-learn

:: 启动 Streamlit
streamlit run app.py

:: 如果程序异常退出，保持窗口打开以便查看错误信息
pause
