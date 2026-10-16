"""The ``python -m riemann`` command-line interface."""

from __future__ import annotations

from riemann import cli


def test_cli_runs_and_reports(capsys):
    assert cli.main(["sod", "--nx", "50"]) == 0
    out = capsys.readouterr().out
    assert "p* = 0.30313" in out
    assert "L1" in out

def test_cli_reports_documented_failure(capsys):
    """toro2 must exit non-zero with the explanatory message, not a traceback."""
    assert cli.main(["toro2", "--nx", "50"]) == 1
    captured = capsys.readouterr()
    assert "numerical run FAILED" in captured.err
    assert "positivity" in captured.err or "non-physical" in captured.err

def test_cli_rejects_unknown_problem(capsys):
    """An unknown name gets a listing of the valid ones, not a traceback."""
    assert cli.main(["definitely-not-a-problem"]) == 2
    err = capsys.readouterr().err
    assert "unknown problem" in err
    assert "sod" in err

def test_cli_first_order_and_limiter_flags(capsys):
    assert cli.main(["sod", "--nx", "50", "--first-order", "--limiter", "minmod"]) == 0
    assert "1st order" in capsys.readouterr().out

def test_cli_accepts_readable_flux_spellings(capsys):
    """``--flux lax_friedrichs`` must work, not just the normalised key.

    The flux registry stores normalised keys (no spaces, underscores or
    hyphens), so an argparse ``choices`` list built from it would have
    rejected the spelling the help text and the docs use.
    """
    for spelling in ("lax_friedrichs", "lf", "hlle", "llf", "HLLC"):
        assert cli.main(["sod", "--nx", "50", "--flux", spelling]) == 0
        capsys.readouterr()

def test_cli_rejects_unknown_flux(capsys):
    assert cli.main(["sod", "--flux", "bogus"]) == 2
    assert "unknown flux" in capsys.readouterr().err

def test_cli_reports_flux_specific_failure(capsys):
    """A second-order failure names the TVD correction as the likely cause.

    AUSM+-up runs every built-in problem at first order; `toro3` is one of the
    three it cannot take at second order with the characteristic-based
    limiter, so the message should point at that rather than at the flux.
    """
    assert cli.main(["toro3", "--nx", "100", "--flux", "ausmup"]) == 1
    err = capsys.readouterr().err
    assert "TVD correction" in err
    assert "second_order=False" in err
    assert "hll" in err  # points at the fluxes that do run everything
