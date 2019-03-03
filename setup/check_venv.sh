#!/bin/bash

# Returns 0 if not in venv, 1 if in venv
exec python -c 'import sys; sys.exit(sys.prefix.endswith("venv"))'
