"""
CLI entry point for offarm.

This module enables running offarm as a Python module:
    $ python -m offarm run path/to/manifest.json
    $ python -m offarm --version
"""

from .cli import main

if __name__ == "__main__":
    main()
