#!/usr/bin/env python3
"""
Register Baseline Theme extension in Cursor's extensions.json
"""
import json
import os
import uuid
from pathlib import Path

CURSOR_EXT_DIR = Path.home() / ".cursor" / "extensions"
EXTENSIONS_JSON = CURSOR_EXT_DIR / "extensions.json"
EXTENSION_ID = "baseline.baseline-theme"
EXTENSION_VERSION = "1.0.0"
EXTENSION_PATH = CURSOR_EXT_DIR / EXTENSION_ID

def main():
    # Check if extension directory exists
    if not EXTENSION_PATH.exists():
        print(f"❌ Extension directory not found: {EXTENSION_PATH}")
        print("Run 'make install' first")
        return 1
    
    # Read existing extensions.json
    if EXTENSIONS_JSON.exists():
        with open(EXTENSIONS_JSON, 'r') as f:
            extensions = json.load(f)
    else:
        extensions = []
    
    # Remove any existing baseline theme entries
    extensions = [e for e in extensions if not (
        isinstance(e, dict) and 
        e.get('identifier', {}).get('id') == EXTENSION_ID
    )]
    
    # Generate UUID for this extension
    extension_uuid = str(uuid.uuid4())
    publisher_uuid = str(uuid.uuid4())
    
    # Create new entry
    new_entry = {
        "identifier": {
            "id": EXTENSION_ID,
            "uuid": extension_uuid
        },
        "version": EXTENSION_VERSION,
        "location": {
            "$mid": 1,
            "path": str(EXTENSION_PATH),
            "scheme": "file"
        },
        "relativeLocation": EXTENSION_ID,
        "metadata": {
            "id": extension_uuid,
            "publisherId": publisher_uuid,
            "publisherDisplayName": "baseline",
            "targetPlatform": "undefined",
            "isApplicationScoped": False,
            "updated": False,
            "isPreReleaseVersion": False,
            "installedTimestamp": int(Path(EXTENSION_PATH).stat().st_mtime * 1000)
        }
    }
    
    # Add to list
    extensions.append(new_entry)
    
    # Write back
    with open(EXTENSIONS_JSON, 'w') as f:
        json.dump(extensions, f, indent=2)
    
    print("✅ Registered extension in extensions.json")
    print(f"Extension ID: {EXTENSION_ID}")
    print(f"UUID: {extension_uuid}")
    return 0

if __name__ == "__main__":
    exit(main())

