"""Allow ``python -m src`` to invoke the command-line interface."""

from .cli import main

raise SystemExit(main())
