贡献指南
========

我们欢迎对 Riemann 库的贡献！本文档提供了为项目做贡献的指南。

如何贡献
--------

有多种方式可以为 Riemann 做出贡献：

1. **报告错误**：如果您发现了错误，请在我们的 Gitee 仓库上创建 issue 来报告它。
2. **建议功能**：有新功能的想法？请打开 issue 来讨论它。
3. **提交 Pull Request**：如果您想贡献代码，请按照以下步骤操作。

开发环境设置
------------

1. 在 Gitee 上 fork 仓库
2. 在本地克隆您的 fork：

   .. code-block:: bash

      git clone https://gitee.com/[your-username]/Riemann.git
      cd Riemann

3. 创建虚拟环境：

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # 在 Windows 上: venv\Scripts\activate

4. 以开发模式安装：

   .. code-block:: bash

      pip install -e .

5. 安装测试依赖：

   .. code-block:: bash

      pip install -e .[tests]

代码风格
--------

- 遵循 Python 的 PEP 8 代码风格
- 使用有意义的变量和函数名
- 为所有公共函数和类添加文档字符串
- 在提交 pull request 之前确保所有测试通过

提交更改
--------

1. 为您的功能或错误修复创建一个新分支：

   .. code-block:: bash

      git checkout -b feature-name

2. 进行更改并在适用时添加测试
3. 运行测试：

   .. code-block:: bash

      pytest

4. 提交您的更改：

   .. code-block:: bash

      git commit -m "您的更改描述"

5. 推送到您的 fork：

   .. code-block:: bash

      git push origin feature-name

6. 在 Gitee 上创建 pull request

许可证
~~~~~~

通过为 Riemann 做出贡献，您同意您的贡献将根据 BSD-3-Clause 许可证进行许可。