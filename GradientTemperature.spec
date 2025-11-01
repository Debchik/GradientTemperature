# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

project_root = (
    Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
)


def collect_data_dir(relative_path: str) -> list[tuple[str, str]]:
    base_path = project_root / relative_path
    if not base_path.exists():
        return []
    files = []
    for file_path in base_path.rglob('*'):
        if file_path.is_file():
            dest = Path(relative_path) / file_path.relative_to(base_path)
            files.append((str(file_path), str(dest)))
    return files


datas = [
    (str(project_root / 'config.json'), 'config.json'),
]
datas += collect_data_dir('languages')
datas += collect_data_dir('_internal')


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GradientTemperature',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='GradientTemperature',
)
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='GradientTemperature.app',
        icon=None,
        bundle_identifier=None,
    )
