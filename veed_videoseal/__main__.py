"""Enable running the CLI as a module: ``python -m veed_videoseal <sign|verify> ...``.

Equivalent to the installed ``veed-videoseal`` console script; handy when the package is on
the path but its entry point hasn't been installed (e.g. running straight from a checkout).
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
