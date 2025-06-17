#!/usr/bin/env python3
import subprocess
import sys

try:
    result = subprocess.run(['python3', '/home/ubuntu/blackletter/pipeline_runner.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error running pipeline: {e}", file=sys.stderr)
    sys.exit(1)
