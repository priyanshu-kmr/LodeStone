@echo off
rem Starts LodeStone without leaving a console window open.
start "" "%~dp0.venv\Scripts\pythonw.exe" -m lodestone.ui.app
