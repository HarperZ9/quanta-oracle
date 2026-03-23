"""
Quanta Oracle — GUI Package

Launch the PyQt6 application window.
"""


def launch():
    import sys
    from PyQt6.QtWidgets import QApplication
    from quanta_oracle.gui.app import QuantaOracleWindow

    app = QApplication(sys.argv)
    window = QuantaOracleWindow()
    window.show()
    return app.exec()
