# Minimal makefile for Sphinx documentation (with multi-language support)
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
SOURCEDIR_ZH  = source_zh
BUILDDIR_ZH   = build/html_zh

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Build English documentation
html-en:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "English documentation built at $(BUILDDIR)/html"

# Build Chinese documentation
html-zh:
	@cp $(SOURCEDIR_ZH)/conf_zh.py $(SOURCEDIR_ZH)/conf.py
	@$(SPHINXBUILD) -b html "$(SOURCEDIR_ZH)" "$(BUILDDIR_ZH)" -c "$(SOURCEDIR_ZH)" $(SPHINXOPTS) $(O)
	@rm $(SOURCEDIR_ZH)/conf.py
	@echo "Chinese documentation built at $(BUILDDIR_ZH)"

# Build both English and Chinese documentation
html-all:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo "English documentation built at $(BUILDDIR)/html"
	@cp $(SOURCEDIR_ZH)/conf_zh.py $(SOURCEDIR_ZH)/conf.py
	@$(SPHINXBUILD) -b html "$(SOURCEDIR_ZH)" "$(BUILDDIR_ZH)" -c "$(SOURCEDIR_ZH)" $(SPHINXOPTS) $(O)
	@rm $(SOURCEDIR_ZH)/conf.py
	@echo "Chinese documentation built at $(BUILDDIR_ZH)"

# Clean all build files
clean-all:
	@rm -rf $(BUILDDIR)/*
	@echo "All build files cleaned"

# Default to building English documentation (maintains backward compatibility)
html: html-en

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)