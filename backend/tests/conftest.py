import sys
import os

# Make backend/ importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
