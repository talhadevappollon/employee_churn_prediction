@echo off
echo ========================================
echo  Statistical Analysis Streamlit App
echo ========================================
echo.
echo Starting the app...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
python -m streamlit run statistical_analysis_app.py

pause
