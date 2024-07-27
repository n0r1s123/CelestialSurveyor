# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import copy_metadata

datas = []
datas += copy_metadata('xisf')
datas += copy_metadata('twirl')
datas += copy_metadata('astroquery')
datas += copy_metadata('tensorflow')

a = Analysis(
    ['main.py'],
    pathex=['C:\\git\\object_recognition\\'],
    binaries=[('.\\model\\model161.bin', '.\\lib-dynload\\'), ('.\\backend\\astroquery\\CITATION', '.\\astroquery')],
    datas=datas,
    hiddenimports=['astroquery'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CelestialSurveyor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CelestialSurveyor',
)