# PhysioMotion4D Logging API Reference

## Quick Reference

All PhysioMotion4D classes share a single logger called "PhysioMotion4D" with class names shown in brackets.

### Class Methods (Global Control)

```python
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase
import logging

# Control log level for ALL classes
PhysioMotion4DBase.set_log_level(logging.INFO)
PhysioMotion4DBase.set_log_level('DEBUG')  # Can use string too

# Filter to show only specific classes
PhysioMotion4DBase.set_log_classes(["RegisterModelsPCA", "WorkflowRegisterHeartModelToPatient"])

# Show all classes (disable filtering)
PhysioMotion4DBase.set_log_all_classes()

# Query which classes are currently filtered
classes = PhysioMotion4DBase.get_log_classes()
print(classes)  # [] if all enabled, or list of filtered class names
```

### Instance Methods (Per-Object Logging)

```python
# Basic logging methods
obj.log_debug("Detailed diagnostic information")
obj.log_info("General informational message")
obj.log_warning("Warning message")
obj.log_error("Error message")
obj.log_critical("Critical error message")

# Convenience methods
obj.log_section("Section Title", width=70, char='=')
obj.log_progress(current, total, prefix="Processing")
```

## Complete API

### PhysioMotion4DBase Class Methods

| Method                         | Parameters               | Description                            |
| ------------------------------ | ------------------------ | -------------------------------------- |
| `set_log_level(log_level)`     | `log_level: int \| str`  | Set logging level for all classes      |
| `set_log_classes(class_names)` | `class_names: list[str]` | Show logs only from specified classes  |
| `set_log_all_classes()`        | None                     | Show logs from all classes             |
| `get_log_classes()`            | None                     | Get list of currently filtered classes |

### Instance Methods

| Method                                 | Parameters                                         | Description                  |
| -------------------------------------- | -------------------------------------------------- | ---------------------------- |
| `log_debug(message)`                   | `message: str`                                     | Log DEBUG level message      |
| `log_info(message)`                    | `message: str`                                     | Log INFO level message       |
| `log_warning(message)`                 | `message: str`                                     | Log WARNING level message    |
| `log_error(message)`                   | `message: str`                                     | Log ERROR level message      |
| `log_critical(message)`                | `message: str`                                     | Log CRITICAL level message   |
| `log_section(title, width, char)`      | `title: str, width: int=70, char: str='='`         | Log formatted section header |
| `log_progress(current, total, prefix)` | `current: int, total: int, prefix: str='Progress'` | Log progress information     |

## Usage Patterns

### Pattern 1: Basic Class with Logging
```python
import logging
from physiomotion4d.physiomotion4d_base import PhysioMotion4DBase

class MyProcessor(PhysioMotion4DBase):
    def __init__(self, log_level=logging.INFO):
        super().__init__(logger_name="MyProcessor", log_level=log_level)
    
    def process(self):
        self.log_info("Starting processing...")
        self.log_debug("Detailed debug info")
        self.log_info("Processing complete!")
```

### Pattern 2: Using Section Headers
```python
def run_workflow(self):
    self.log_section("Stage 1: Initialization")
    self.log_info("Initializing...")
    
    self.log_section("Stage 2: Processing")
    self.log_info("Processing...")
    
    self.log_section("Complete")
```

### Pattern 3: Progress Reporting
```python
def process_items(self, items):
    n_items = len(items)
    progress_interval = max(1, n_items // 10)
    
    for i, item in enumerate(items):
        if i % progress_interval == 0 or i == n_items - 1:
            self.log_progress(i + 1, n_items, prefix="Processing items")
        
        # ... process item ...
```

### Pattern 4: Global Log Control
```python
# Create multiple objects
pca_reg = RegisterModelsPCA(..., log_level=logging.INFO)
workflow = WorkflowRegisterHeartModelToPatient(..., log_level=logging.INFO)

# Change log level for both at once
PhysioMotion4DBase.set_log_level(logging.DEBUG)

# Both now show debug messages
pca_reg.log_debug("Debug from PCA")
workflow.log_debug("Debug from workflow")

# Switch to quiet mode
PhysioMotion4DBase.set_log_level(logging.WARNING)
```

### Pattern 5: Selective Class Filtering
```python
# Show only PCA registration logs
PhysioMotion4DBase.set_log_classes(["RegisterModelsPCA"])

pca_reg.log_info("This is shown")
workflow.log_info("This is hidden")

# Show both classes
PhysioMotion4DBase.set_log_classes([
    "RegisterModelsPCA",
    "WorkflowRegisterHeartModelToPatient"
])

# Show all classes
PhysioMotion4DBase.set_log_all_classes()
```

### Pattern 6: Debugging Specific Components
```python
# Start with all classes at INFO level
PhysioMotion4DBase.set_log_level(logging.INFO)
PhysioMotion4DBase.set_log_all_classes()

# Run initial processing
workflow.run_workflow()

# Focus on specific class for debugging
PhysioMotion4DBase.set_log_level(logging.DEBUG)
PhysioMotion4DBase.set_log_classes(["RegisterModelsPCA"])

# Now only PCA registration shows detailed debug output
workflow.run_workflow()

# Back to normal
PhysioMotion4DBase.set_log_level(logging.INFO)
PhysioMotion4DBase.set_log_all_classes()
```

## Log Levels

| Level              | Numeric Value | When to Use                                   |
| ------------------ | ------------- | --------------------------------------------- |
| `logging.DEBUG`    | 10            | Detailed diagnostic information for debugging |
| `logging.INFO`     | 20            | General informational messages (default)      |
| `logging.WARNING`  | 30            | Warning messages about potential issues       |
| `logging.ERROR`    | 40            | Error messages for serious problems           |
| `logging.CRITICAL` | 50            | Critical errors that may cause termination    |

## Output Format

All log messages follow this format:
```
TIMESTAMP - PhysioMotion4D - LEVEL - [ClassName] Message
```

Example:
```
2025-12-13 11:35:27 - PhysioMotion4D - INFO - [RegisterModelsPCA] Converting mean shape points...
2025-12-13 11:35:27 - PhysioMotion4D - DEBUG - [WorkflowRegisterHeartModelToPatient] Auto-generating masks...
2025-12-13 11:35:27 - PhysioMotion4D - WARNING - [RegisterModelsPCA] No points found within threshold
```

## Available Classes

Current PhysioMotion4D classes with logging support:
- `RegisterModelsPCA` - PCA-based model-to-image registration
- `WorkflowRegisterHeartModelToPatient` - Multi-stage heart model registration
- (More classes will be added as they are converted)

## Common Use Cases

### Use Case 1: Normal Operation
```python
# Default: INFO level, all classes shown
registrar = RegisterModelsPCA(..., log_level=logging.INFO)
result = registrar.register(...)
```

### Use Case 2: Quiet Mode (Minimal Output)
```python
# WARNING level: only warnings and errors
PhysioMotion4DBase.set_log_level(logging.WARNING)
registrar = RegisterModelsPCA(...)
result = registrar.register(...)
```

### Use Case 3: Debug Specific Component
```python
# Debug only RegisterModelsPCA
PhysioMotion4DBase.set_log_level(logging.DEBUG)
PhysioMotion4DBase.set_log_classes(["RegisterModelsPCA"])

workflow = WorkflowRegisterHeartModelToPatient(...)
# Only PCA component will show debug messages
workflow.run_workflow()
```

### Use Case 4: Production Logging to File
```python
# Log everything to file for analysis
registrar = RegisterModelsPCA(
    ...,
    log_level=logging.DEBUG,
    log_to_file="registration.log"
)
```

### Use Case 5: Progressive Debugging
```python
# Start at INFO level
PhysioMotion4DBase.set_log_level(logging.INFO)

# If something goes wrong, switch to DEBUG
PhysioMotion4DBase.set_log_level(logging.DEBUG)

# Focus on specific class
PhysioMotion4DBase.set_log_classes(["RegisterModelsPCA"])

# Re-run the problematic part
registrar.optimize_rigid_alignment(...)
```

## Tips and Best Practices

1. **Use INFO for normal operations** - This is the default and provides good visibility
2. **Use DEBUG for troubleshooting** - Detailed information when things go wrong
3. **Use WARNING for quiet mode** - Only see problems, not routine messages
4. **Filter classes for complex workflows** - Focus on the component of interest
5. **Use log_section() for major stages** - Makes output easier to read
6. **Report progress for long operations** - Keep users informed during lengthy processes

## Thread Safety

⚠️ **Note**: The current implementation is not thread-safe. If using multiple threads:
- Set log level and class filters before creating threads
- Avoid changing logging configuration while threads are running
- Consider using separate logger instances for multi-threaded applications

## Performance Considerations

- The shared logger is efficient and has minimal overhead
- Class filtering adds negligible performance cost
- Progress reporting should be used judiciously (every 10% is reasonable)
- DEBUG level messages have minimal cost when DEBUG is not enabled

