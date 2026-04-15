# Read the Docs 多语言文档配置说明

本项目支持中英文双语文档，通过 Read the Docs 的**子项目**功能实现。

## 项目结构

```
docs/
├── source/          # 英文文档源文件
│   ├── conf.py     # 英文 Sphinx 配置
│   └── *.rst       # 英文文档页面
├── source_zh/       # 中文文档源文件
│   ├── conf.py     # 中文 Sphinx 配置 (language = 'zh_CN')
│   └── *.rst       # 中文文档页面
├── requirements.txt # 文档构建依赖
├── Makefile
└── make.bat
```

## 配置文件

### 1. 英文文档配置 (.readthedocs.yml)

用于构建英文文档，配置指向 `docs/source/conf.py`。

### 2. 中文文档配置 (.readthedocs.zh.yml)

用于构建中文文档，配置指向 `docs/source_zh/conf.py`。

## Read the Docs 设置步骤

### 步骤 1: 创建主项目（英文）

1. 登录 [Read the Docs](https://readthedocs.org/)
2. 导入项目，选择你的 Git 仓库
3. 项目名称建议: `riemann` 或 `riemann-en`
4. 配置文件路径: `.readthedocs.yml`（默认）
5. 保存并构建

### 步骤 2: 创建子项目（中文）

1. 在 Read the Docs 中再次导入同一个 Git 仓库
2. 项目名称建议: `riemann-zh`
3. **关键设置**: 在项目的 **Advanced Settings** 中：
   - **Configuration file**: 填写 `.readthedocs.zh.yml`
4. 保存并构建

### 步骤 3: 设置子项目关联

1. 进入英文主项目的 **Admin** → **Subprojects**
2. 添加子项目：
   - **Child**: 选择 `riemann-zh`
   - **Alias**: 填写 `zh-cn` 或 `zh`
3. 保存

### 步骤 4: 配置语言切换

在子项目设置中：
- **Language**: 选择 `Chinese (Simplified, China)`

## 访问地址

配置完成后，文档将通过以下地址访问：

- **英文文档**: `https://riemann.readthedocs.io/en/latest/`
- **中文文档**: `https://riemann.readthedocs.io/zh-cn/latest/`

或如果分别设置项目：
- **英文**: `https://riemann-en.readthedocs.io/en/latest/`
- **中文**: `https://riemann-zh.readthedocs.io/zh-cn/latest/`

## 本地测试

### 构建英文文档

```bash
cd docs
make html
# 输出在 build/html/
```

### 构建中文文档

```bash
cd docs
sphinx-build -b html source_zh build/html_zh
# 输出在 build/html_zh/
```

## 自动构建

每次推送到 Git 仓库的默认分支（如 main/master）时，Read the Docs 会自动：
1. 检出最新代码
2. 安装依赖 (`docs/requirements.txt`)
3. 构建文档
4. 发布到对应 URL

## 注意事项

1. **中文搜索**: 已配置 `jieba` 分词库支持中文搜索
2. **子项目同步**: 子项目会继承主项目的部分设置，但配置文件独立
3. **版本管理**: 不同语言文档的版本可以独立管理
4. **跨语言链接**: 首页已添加语言切换链接

## 故障排除

### 中文文档构建失败

检查 `docs/source_zh/conf.py` 中的 `language = 'zh_CN'` 是否正确设置。

### 搜索功能不正常

确保 `docs/requirements.txt` 中包含 `jieba` 依赖。

### 样式问题

中文文档可能需要额外的 CSS 调整，可在 `docs/source_zh/_static/` 中添加自定义样式。
