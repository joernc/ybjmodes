Get modes, for example, with
```julia
λ, ϕ, a = ybjmodes(dipole, 512, 0.125, "mod"; adv=true, order=4, nev=400);
```
This is requesting the first 400 modes for the dipole mean flow with 512 by 512 resolution, ε = 1/8, saved with the tag `mod`. Advection is turned on, and fourth-order finite differences are used. If these modes have been previously calculated, they are loaded from a file in `/groups/oceanphysics/ybjmodes/` (change this path if you do not work on the Caltech HPC).
