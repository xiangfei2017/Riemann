# Multi-language Documentation Build Guide

This project supports bilingual documentation in Chinese and English, using the Sphinx build system.

## Directory Structure

```
docs/
├── source/           # English documentation source files
│   ├── conf.py       # English documentation configuration
│   ├── index.rst     # English documentation homepage
│   └── ...
├── source_zh/        # Chinese documentation source files
│   ├── conf_zh.py    # Chinese documentation configuration
│   ├── index.rst     # Chinese documentation homepage
│   └── ...
├── build/
│   ├── html/         # English documentation build output
│   └── html_zh/      # Chinese documentation build output
├── Makefile          # Linux/Mac build scripts
└── make.bat          # Windows build scripts
```

## Build Commands

### Using Makefile (Linux/Mac)

```bash
# Build English documentation
make html-en

# Build Chinese documentation
make html-zh

# Build all language documentation
make html-all

# Clean all build files
make clean-all
```

### Using Batch Files (Windows)

```cmd
# Build English documentation
make.bat html-en

# Build Chinese documentation
make.bat html-zh

# Build all language documentation
make.bat html-all

# Clean all build files
make.bat clean-all
```

## Configuration Details

### English Documentation Configuration
- Configuration file: `source/conf.py`
- Language setting: Default (English)
- Output directory: `build/html/`

### Chinese Documentation Configuration
- Configuration file: `source_zh/conf.py`
- Language setting: `language = 'zh_CN'`
- Output directory: `build/html_zh/`

## Adding New Documentation

### English Documentation
1. Add new `.rst` files in the `source/` directory
2. Add to toctree in `source/index.rst`

### Chinese Documentation
1. Add new `.rst` files in the `source_zh/` directory
2. Add to toctree in `source_zh/index.rst`

## Important Notes

1. Chinese documentation requires correct encoding and language configuration
2. Ensure the system has installed Sphinx themes that support Chinese
3. Specify the use of `source_zh/conf.py` configuration file when building Chinese documentation