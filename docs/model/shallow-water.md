# Shallow-water model

The shallow-water model describes a shallow layer over a flat frictionless bed in one
dimension. Inputs use `[h, u]`, with nonnegative depth `h`, depth-averaged velocity `u`,
and positive gravity.

## The equations

For a layer of fluid whose depth is small compared with the wavelength of the motion,
over a flat frictionless bed, conservation of mass and momentum gives

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = \mathbf{0},
$$

$$
\mathbf{U} = [h, hu]^{T}, \qquad \mathbf{F} = [hu, hu^{2} + \tfrac{1}{2}gh^{2}]^{T},
$$

with $h$ the depth, $u$ the depth-averaged velocity and $g$ gravity.

There is no equation of state to supply and no energy equation: the $\tfrac{1}{2}gh^{2}$
term is the depth-integrated hydrostatic pressure, and the system closes itself. That
makes it the simplest non-trivial non-linear hyperbolic system, and the natural second
system for a code built around the Euler equations.

The Jacobian has eigenvalues

$$
\lambda_{1} = u - a, \qquad \lambda_{2} = u + a, \qquad a = \sqrt{gh},
$$

with $a$ the celerity of a small gravity wave. The Froude number $u/a$ plays the role of
the Mach number: subcritical and supercritical correspond to subsonic and supersonic.

______________________________________________________________________

Depth has length units, velocity has length/time units, and gravity has length/time²
units. Standard examples use metres and seconds. Bed topography, friction, and an
independent energy equation are outside this model. See [parameters](parameters.md) and
[dry-bed workflows](../running/shallow-water.md).
