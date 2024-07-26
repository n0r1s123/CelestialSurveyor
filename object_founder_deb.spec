# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import copy_metadata

datas = []
datas += copy_metadata('xisf')
datas += [('/home/n0r1s/miniconda3/envs/CelestialSurveyor/lib/libssl.so.3', '.')]
datas += [('/home/n0r1s/miniconda3/envs/CelestialSurveyor/lib/libcrypto.so.3', '.')]
print(datas)

a = Analysis(
    ['main.py'],
    pathex=['/home/n0r1s/git/CelestialSurveyor'],
    binaries=[('./model/model161.bin', './lib-dynload/'), ('./backend/astroquery/CITATION', './astroquery')],
    datas=datas,
    hiddenimports=['astroquery', 'scipy._lib.array_api_compat.numpy.fft', 'scipy.special._special_ufuncs', 'ssl', 'urllib.request.http.client'],
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