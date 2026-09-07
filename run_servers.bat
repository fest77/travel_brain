@echo off
rem ============================================================
rem  travel_brain 一键启动：导入服务(8000) + 查询服务(8001)
rem  双击本文件，会弹出两个服务窗口；关闭窗口即停止服务。
rem  提示：首次提问会加载本地模型，需等待十几秒~半分钟。
rem ============================================================
cd /d "%~dp0"

start "travel-import-8000" cmd /k ".venv\Scripts\python.exe knowledge\api\import_router.py"
start "travel-query-8001"  cmd /k ".venv\Scripts\python.exe knowledge\api\query_router.py"

echo.
echo Services are starting, wait ~30-40s then open:
echo   Import UI : http://127.0.0.1:8000/import
echo   Chat   UI : http://127.0.0.1:8001/chat
echo.
