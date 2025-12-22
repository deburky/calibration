# Test File for Baseline Theme Preview

Open this file in VS Code to preview the theme colors.

## Python Example

```python
from loguru import logger
from rich.logging import RichHandler
import sys

def setup_logger(level: str = "INFO", var=50):
    """Set up a logger with RichHandler for better formatting and color support."""
    logger.remove()  # Ensure no duplicated logs
    logger.add(sys.stdout, format="{message}")
    logger.configure(
        handlers=[{"sink": RichHandler(), "format": "{message}", "level": level}]
    )
    return logger

setup_logger()
```

## JavaScript Example

```javascript
// This is a comment
function setupLogger(level = "INFO") {
    const logger = console;
    logger.info(`Logger initialized with level: ${level}`);
    return logger;
}

const myLogger = setupLogger("DEBUG");
const config = {
    level: "INFO",
    format: "{message}"
};
```

## TypeScript Example

```typescript
interface LoggerConfig {
    level: string;
    format: string;
}

class Logger {
    private config: LoggerConfig;
    
    constructor(config: LoggerConfig) {
        this.config = config;
    }
    
    log(message: string): void {
        console.log(message);
    }
}

const logger = new Logger({ level: "INFO", format: "{message}" });
```

## HTML Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a paragraph with <strong>bold</strong> text.</p>
</body>
</html>
```

## CSS Example

```css
/* This is a CSS comment */
body {
    background-color: #1e1e1e;
    color: #abb2bf;
    font-family: 'Inter', sans-serif;
}

.button {
    background: #528bff;
    color: #ffffff;
    padding: 8px 16px;
}
```









