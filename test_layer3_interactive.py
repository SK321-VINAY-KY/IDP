"""
test_layer3_interactive.py — Root launcher for Layer 3 extraction & interactive chatbot.

Usage:
    python test_layer3_interactive.py
    python test_layer3_interactive.py --md tests/fixtures/MY_resume.md --schema tests/schemas/schema.json
    python test_layer3_interactive.py --md sdg_goals_output --schema schema --query "What is the first goal?"
    pytest tests/test_layer3_pipeline.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA_APP = ROOT / "schema_chatbot_v2"
if str(SCHEMA_APP) not in sys.path:
    sys.path.insert(0, str(SCHEMA_APP))

from tests.test_layer3_pipeline import main

if __name__ == "__main__":
    main()
