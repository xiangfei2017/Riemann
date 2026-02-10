@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation (with multi-language support)

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR_EN=source
set SOURCEDIR_ZH=source_zh
set BUILDDIR=build
set BUILDDIR_EN=%BUILDDIR%\html
set BUILDDIR_ZH=%BUILDDIR%\html_zh

if "%1" == "" goto help
if "%1" == "html-en" goto html_en
if "%1" == "html-zh" goto html_zh
if "%1" == "html-all" goto html_all
if "%1" == "clean-all" goto clean_all

%SPHINXBUILD% >nul 2>nul
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR_EN% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html_en
%SPHINXBUILD% -b html %SOURCEDIR_EN% %BUILDDIR_EN% %SPHINXOPTS% %O%
echo English documentation built at %BUILDDIR_EN%
goto end

:html_zh
%SPHINXBUILD% -b html %SOURCEDIR_ZH% %BUILDDIR_ZH% -c %SOURCEDIR_ZH% %SPHINXOPTS% %O%
echo Chinese documentation built at %BUILDDIR_ZH%
goto end

:html_all
%SPHINXBUILD% -b html %SOURCEDIR_EN% %BUILDDIR_EN% %SPHINXOPTS% %O%
echo English documentation built at %BUILDDIR_EN%
%SPHINXBUILD% -b html %SOURCEDIR_ZH% %BUILDDIR_ZH% -c %SOURCEDIR_ZH% %SPHINXOPTS% %O%
echo Chinese documentation built at %BUILDDIR_ZH%
goto end

:clean_all
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
echo All build files cleaned
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR_EN% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd