Contributing
============

We welcome contributions to the Riemann library! This document provides guidelines for contributing to the project.

How to Contribute
-----------------

There are several ways to contribute to Riemann:

1. **Report Bugs**: If you find a bug, please report it by creating an issue on our Gitee repository.
2. **Suggest Features**: Have an idea for a new feature? Please open an issue to discuss it.
3. **Submit Pull Requests**: If you'd like to contribute code, please follow the steps below.

Development Setup
-----------------

1. Fork the repository on Gitee
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://gitee.com/[your-username]/Riemann.git
      cd Riemann

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

   .. code-block:: bash

      pip install -e .

5. Install test dependencies:

   .. code-block:: bash

      pip install -e .[tests]

Code Style
----------

- Follow PEP 8 for Python code style
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Ensure all tests pass before submitting a pull request

Submitting Changes
------------------

1. Create a new branch for your feature or bugfix:

   .. code-block:: bash

      git checkout -b feature-name

2. Make your changes and add tests if applicable
3. Run the tests:

   .. code-block:: bash

      pytest

4. Commit your changes:

   .. code-block:: bash

      git commit -m "Description of your changes"

5. Push to your fork:

   .. code-block:: bash

      git push origin feature-name

6. Create a pull request on Gitee

License
-------

By contributing to Riemann, you agree that your contributions will be licensed under the BSD-3-Clause License.