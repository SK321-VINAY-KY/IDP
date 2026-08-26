import os
import sys
import runpy

# Ensure project root is importable and propagate stored key
project_root = os.path.abspath(os.path.dirname(__file__) + "\\..")
sys.path.insert(0, project_root)
os.environ['GOOGLE_API_KEY'] = os.environ.get('IDP_GEMINI_API_KEY', '')

runpy.run_path('scripts/run_escalation_test.py', run_name='__main__')
