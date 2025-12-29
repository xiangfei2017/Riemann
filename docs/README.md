# 多语言文档构建指南

本项目支持中英文双语文档，使用 Sphinx 构建系统。

## 目录结构

```
docs/
├── source/           # 英文文档源文件
│   ├── conf.py       # 英文文档配置
│   ├── index.rst     # 英文文档首页
│   └── ...
├── source_zh/        # 中文文档源文件
│   ├── conf_zh.py    # 中文文档配置
│   ├── index.rst     # 中文文档首页
│   └── ...
├── build/
│   ├── html/         # 英文文档构建输出
│   └── html_zh/      # 中文文档构建输出
├── Makefile          # Linux/Mac 构建脚本
└── make.bat          # Windows 构建脚本
```

## 构建命令

### 使用 Makefile (Linux/Mac)

```bash
# 构建英文文档
make html-en

# 构建中文文档
make html-zh

# 构建所有语言文档
make html-all

# 清理所有构建文件
make clean-all
```

### 使用批处理文件 (Windows)

```cmd
# 构建英文文档
make.bat html-en

# 构建中文文档
make.bat html-zh

# 构建所有语言文档
make.bat html-all

# 清理所有构建文件
make.bat clean-all
```

## 配置说明

### 英文文档配置
- 配置文件：`source/conf.py`
- 语言设置：默认（英语）
- 输出目录：`build/html/`

### 中文文档配置
- 配置文件：`source_zh/conf_zh.py`
- 语言设置：`language = 'zh_CN'`
- 输出目录：`build/html_zh/`

## 添加新文档

### 英文文档
1. 在 `source/` 目录下添加新的 `.rst` 文件
2. 在 `source/index.rst` 中添加到 toctree

### 中文文档
1. 在 `source_zh/` 目录下添加新的 `.rst` 文件
2. 在 `source_zh/index.rst` 中添加到 toctree

## 注意事项

1. 中文文档需要设置正确的编码和语言配置
2. 确保系统已安装支持中文的 Sphinx 主题
3. 构建中文文档时指定使用 `source_zh/conf_zh.py` 配置文件