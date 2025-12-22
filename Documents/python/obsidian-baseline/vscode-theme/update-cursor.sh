#!/bin/bash

echo "Updating Baseline Theme in Cursor..."
echo ""

cd ~/.cursor/extensions

# Remove old
rm -rf baseline-theme baseline.baseline-theme

# Install fresh
mkdir -p baseline.baseline-theme
cp /Users/deburky/Documents/python/obsidian-baseline/vscode-theme/package.json baseline.baseline-theme/
cp -r /Users/deburky/Documents/python/obsidian-baseline/vscode-theme/themes baseline.baseline-theme/

echo "✅ Theme updated!"
echo ""
echo "Now:"
echo "1. Quit Cursor (Cmd+Q)"
echo "2. Reopen Cursor"
echo "3. Open a Python file to see the changes"
echo "4. If colors don't update, press Cmd+Shift+P → 'Developer: Reload Window'"





























































