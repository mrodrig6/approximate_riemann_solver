"""Test package.

The directories mirror ``src/``: ``core/``, ``euler/`` and
``shallow_water/``, with ``test_cli.py`` at the top beside the module it
covers.  The ``__init__.py`` files are what let ``euler/test_exact.py`` and
``shallow_water/test_exact.py`` share a basename without pytest mistaking
one for the other.
"""
