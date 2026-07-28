# conftest.py — shared pytest configuration for MindTrack tests
# Ensures the project root is on sys.path for all test modules.

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
