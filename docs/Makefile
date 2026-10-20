SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

html doctest linkcheck:
	$(SPHINXBUILD) -b $@ $(SOURCEDIR) $(BUILDDIR)/$@

clean:
	rm -rf $(BUILDDIR)

.PHONY: html doctest linkcheck clean
