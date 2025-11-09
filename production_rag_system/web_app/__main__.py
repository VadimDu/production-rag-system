"""
Main entry point for the Production RAG System web app when run as a module.

Usage:
    python -m production_rag_system.web_app
    streamlit run production_rag_system/web_app/app.py
"""

import sys
import os

# Add the parent directory to the path so we can import the module
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run the main app
from production_rag_system.web_app.app import main

# Make main function available for entry point
__all__ = ['main']

if __name__ == "__main__":
    main()