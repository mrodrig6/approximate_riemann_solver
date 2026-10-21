# Exact shallow-water solution

The exact solver in `src.shallow_water.exact` solves for the intermediate depth on wet
beds and samples explicit dry-bed branches. These formulas follow Toro, *Shock-Capturing
Methods for Free-Surface Shallow Flows*, Chapters 2 and 5.

## The exact solver

The same construction as the Euler exact solver, with depth in place of pressure. Both
waves pin the middle state, and eliminating $u_{\ast}$ leaves one scalar equation in
$h_{\ast}$:

$$
f(h) = f_{L}(h, h_{L}) + f_{R}(h, h_{R}) + (u_{R} - u_{L}) = 0,
$$

with the one-sided depth functions

$$
f_{K}(h) = 2\left(\sqrt{gh} - \sqrt{gh_{K}}\right) \qquad h \le h_{K} \text{(rarefaction)},
$$

$$
f_{K}(h) = (h - h_{K})\sqrt{\frac{g}{2} \frac{h + h_{K}}{h h_{K}}} \qquad h > h_{K} \text{(bore)}.
$$

$f$ is monotone increasing and concave for wet non-vacuum data. The implementation
guards nonpositive iterates and has a finite iteration cap; monotonicity alone is not an
unconditional convergence guarantee. The starting guess is the two-rarefaction
approximation,

$$
h_{0} = \frac{1}{g}\left[\tfrac{1}{2}(a_{L} + a_{R}) + \tfrac{1}{4}(u_{L} - u_{R})\right]^{2},
$$

which is *exact* when both waves are rarefactions.

### Dry bed

Three dry situations arise, and all three are solved:

| Case             | Condition                            | Solution                                                   |
| ---------------- | ------------------------------------ | ---------------------------------------------------------- |
| Dry to the right | $h_{R} = 0$                          | left rarefaction to a wet/dry front at $u_{L} + 2a_{L}$    |
| Dry to the left  | $h_{L} = 0$                          | mirror image, front at $u_{R} - 2a_{R}$                    |
| Dry generated    | $u_{R} - u_{L} \ge 2(a_{L} + a_{R})$ | both rarefactions run out; a dry region opens between them |

The last condition is the shallow-water counterpart of Toro's vacuum condition for a
gas.

The dry dam break has a closed-form answer — **Ritter's solution** — which makes it a
strong check. Over dry bed the depth at the dam site is $4h_{L}/9$ for all time, the
velocity there is $\tfrac{2}{3}\sqrt{gh_{L}}$, and the wet/dry front advances at
$2\sqrt{gh_{L}}$. The test suite asserts all three.
