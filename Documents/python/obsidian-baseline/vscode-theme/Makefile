.PHONY: install clean verify help redeploy

# Configuration
CURSOR_EXT_DIR := $(HOME)/.cursor/extensions
SOURCE_DIR := $(shell pwd)
TARGET_DIR := $(CURSOR_EXT_DIR)/baseline.baseline-theme

help:
	@echo "Baseline Theme - Makefile Commands"
	@echo ""
	@echo "  make install    - Install theme to Cursor extensions"
	@echo "  make redeploy   - Clean and reinstall theme (recommended)"
	@echo "  make clean      - Remove theme from Cursor extensions"
	@echo "  make verify     - Verify theme installation"
	@echo "  make help       - Show this help message"
	@echo ""

clean:
	@echo "Removing old installations..."
	@rm -rf "$(CURSOR_EXT_DIR)/baseline-theme"
	@rm -rf "$(TARGET_DIR)"
	@echo "✅ Clean complete"

install: clean
	@echo "Installing Baseline Theme to Cursor..."
	@mkdir -p "$(TARGET_DIR)"
	@cp "$(SOURCE_DIR)/package.json" "$(TARGET_DIR)/"
	@cp -r "$(SOURCE_DIR)/themes" "$(TARGET_DIR)/"
	@echo "✅ Installation complete"
	@echo ""
	@echo "NEXT STEPS:"
	@echo "1. Quit Cursor completely (Cmd+Q)"
	@echo "2. Wait 5 seconds"
	@echo "3. Reopen Cursor"
	@echo "4. Check Extensions: Cmd+Shift+X, search '@installed baseline'"
	@echo "5. Or use theme picker: Cmd+K then T, type 'baseline'"

register:
	@echo "Registering extension in Cursor..."
	@python3 "$(SOURCE_DIR)/register-extension.py" || echo "⚠️  Registration failed (may need to run manually)"

redeploy: install register
	@echo ""
	@echo "✅ Theme redeployed and registered successfully!"
	@echo ""
	@echo "NEXT STEPS:"
	@echo "1. Quit Cursor completely (Cmd+Q)"
	@echo "2. Wait 5 seconds"
	@echo "3. Reopen Cursor"
	@echo "4. Check Extensions: Cmd+Shift+X, search '@installed baseline'"
	@echo "5. Or use theme picker: Cmd+K then T, type 'baseline'"

verify:
	@echo "Verifying installation..."
	@if [ -f "$(TARGET_DIR)/package.json" ] && \
	     [ -f "$(TARGET_DIR)/themes/baseline-dark.json" ] && \
	     [ -f "$(TARGET_DIR)/themes/baseline-light.json" ]; then \
		echo "✅ Theme is installed correctly"; \
		echo ""; \
		echo "Installed files:"; \
		ls -la "$(TARGET_DIR)/"; \
		echo ""; \
		echo "Theme files:"; \
		ls -la "$(TARGET_DIR)/themes/"; \
	else \
		echo "❌ Theme is NOT installed"; \
		echo "Run 'make install' or 'make redeploy'"; \
		exit 1; \
	fi

