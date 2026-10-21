# Sod benchmark

This comparison uses the exact Euler solution on the same cell-centred grid as the
numerical result. The textbook check for the exact solver is documented separately in
[Toro reference cases](toro.md).

## Sod's shock tube

The standard first check (Toro Test 1). Left state $(1, 0, 1)$, right state
$(0.125, 0, 0.1)$, $\gamma = 1.4$, output at $t = 0.25$. The solution is a left
rarefaction, a contact and a right shock.

![Sod's problem: exact versus second-order Roe](../figures/sod_comparison.png)

Exact: $p_{\ast} = 0.30313018$, $u_{\ast} = 0.92745262$, reached in 4 Newton iterations.
Toro's Table 4.1 gives $0.30313$ and $0.92745$.

Reading the four panels together is the point of plotting all four. The contact at
$x \approx 0.73$ is plainly visible in $\rho$ and in $e$, and entirely absent from $u$
and $p$ — exactly as the theory requires, since velocity and pressure are continuous
across a linearly degenerate wave. The shock at $x \approx 0.93$ appears in all four.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_sod`.
