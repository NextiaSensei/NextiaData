
#!/usr/bin/env python3
"""
NextiaData Assistant - Punto de entrada principal
"""

import os
import sys

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app import app

if __name__ == '__main__':
    app.run(debug=True)
