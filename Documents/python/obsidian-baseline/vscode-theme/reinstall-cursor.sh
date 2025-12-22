#!/bin/bash

echo "Reinstalling Baseline Theme to Cursor..."
echo ""

# Remove old installation
rm -rf ~/.cursor/extensions/baseline-theme

# Create directory
mkdir -p ~/.cursor/extensions/baseline-theme

# Copy files
cp /Users/deburky/Documents/python/obsidian-baseline/vscode-theme/package.json ~/.cursor/extensions/baseline-theme/
cp -r /Users/deburky/Documents/python/obsidian-baseline/vscode-theme/themes ~/.cursor/extensions/baseline-theme/

echo "✅ Theme reinstalled to Cursor"
echo ""
echo "Next steps:"
echo "1. Quit Cursor completely (Cmd+Q)"
echo "2. Reopen Cursor"
echo "3. Check Extensions: Cmd+Shift+X, search '@installed baseline'"
echo "4. Try theme picker: Cmd+K then T, type 'baseline'"














































