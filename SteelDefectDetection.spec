# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['desktop.py'],
    pathex=[],
    binaries=[],
    datas=[('config.yaml', '.'), ('yolov8n.pt', '.'), ('frontend/dist', 'frontend/dist'), ('src', 'src')],
    hiddenimports=['uvicorn', 'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan', 'uvicorn.lifespan.on', 'fastapi', 'ultralytics', 'yaml', 'pydantic', 'multipart', 'webview', 'webview.platforms.edgechromium', 'webview.platforms.winforms'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['gradio', 'gradio_client', 'tkinter', 'IPython', 'jupyter', 'notebook', 'pytest', 'tensorrt', 'pycuda'],
    noarchive=False,
    module_collection_mode={'torch._numpy': 'py'},
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SteelDefectDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SteelDefectDetection',
)
