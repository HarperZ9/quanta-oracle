import os
block_cipher = None
a = Analysis(
    ['quanta_oracle/cli.py'],
    pathex=[os.path.abspath('.')],
    binaries=[], datas=[],
    hiddenimports=[
        'quanta_oracle', 'quanta_oracle.arima', 'quanta_oracle.prophet',
        'quanta_oracle.changepoint', 'quanta_oracle.decompose', 'quanta_oracle.features',
        'quanta_oracle.metrics', 'quanta_oracle.neural', 'quanta_oracle.autodiff',
        'quanta_oracle.gui', 'quanta_oracle.gui.app',
        'quanta_oracle.gui.pages.dashboard', 'quanta_oracle.gui.pages.forecast_page',
        'quanta_oracle.gui.pages.decompose_page', 'quanta_oracle.gui.pages.changepoint_page',
        'quanta_oracle.gui.pages.settings_page',
        'numpy', 'scipy', 'scipy.optimize', 'scipy.linalg',
        'PyQt6', 'PyQt6.QtWidgets', 'PyQt6.QtCore', 'PyQt6.QtGui',
    ],
    excludes=[], noarchive=False,
)
pyz = PYZ(a.pure, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='quanta-oracle', console=True)
coll = COLLECT(exe, a.binaries, a.datas, name='quanta-oracle')
