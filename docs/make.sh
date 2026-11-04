#!/usr/bin/env bash
# Command file for Sphinx documentation (with multi-language support)
#
# Usage:
#   ./make.sh html-en     Build English HTML docs only
#   ./make.sh html-zh     Build Chinese HTML docs only
#   ./make.sh html-all    Build both English and Chinese HTML docs
#   ./make.sh clean-all   Remove all build artifacts

set -euo pipefail

# cd to the script directory (same as pushd %~dp0 in .bat)
cd "$(dirname "$0")"

if [[ -z "${SPHINXBUILD:-}" ]]; then
    SPHINXBUILD="python -m sphinx"
fi

SOURCEDIR_EN="source"
SOURCEDIR_ZH="source_zh"
BUILDDIR="build"
BUILDDIR_EN="$BUILDDIR/html"
BUILDDIR_ZH="$BUILDDIR/html_zh"

case "${1:-}" in
    "" | "help")
        $SPHINXBUILD -M help $SOURCEDIR_EN $BUILDDIR ${SPHINXOPTS:-} ${O:-}
        ;;

    html-en)
        $SPHINXBUILD -b html $SOURCEDIR_EN $BUILDDIR_EN ${SPHINXOPTS:-} ${O:-}
        echo "English documentation built at $BUILDDIR_EN"
        ;;

    html-zh)
        $SPHINXBUILD -b html $SOURCEDIR_ZH $BUILDDIR_ZH -c $SOURCEDIR_ZH ${SPHINXOPTS:-} ${O:-}
        echo "Chinese documentation built at $BUILDDIR_ZH"
        ;;

    html-all)
        $SPHINXBUILD -b html $SOURCEDIR_EN $BUILDDIR_EN ${SPHINXOPTS:-} ${O:-}
        echo "English documentation built at $BUILDDIR_EN"
        $SPHINXBUILD -b html $SOURCEDIR_ZH $BUILDDIR_ZH -c $SOURCEDIR_ZH ${SPHINXOPTS:-} ${O:-}
        echo "Chinese documentation built at $BUILDDIR_ZH"
        ;;

    clean-all)
        rm -rf "$BUILDDIR"
        echo "All build files cleaned"
        ;;

    *)
        echo "Unknown target: $1"
        echo "Available targets: html-en, html-zh, html-all, clean-all, help"
        exit 1
        ;;
esac
