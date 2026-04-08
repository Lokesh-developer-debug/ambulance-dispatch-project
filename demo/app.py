#!/usr/bin/env python3
"""
Startup script for Hugging Face Spaces deployment.
This is an alternative entry point that can be used by Hugging Face.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from server.app import main
    main()
