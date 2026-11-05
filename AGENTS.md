# Notes for AI agents picking up this repository

Written 2026-11-05 by a Claude Code session working on the JEM coupled model
(`../port-to-new-JEM/case_jcm_veros_topo-realistic-collocated`). Read this
before changing anything.

## Why this exists

The JEM coupled model (JCM SPEEDY T31 atmosphere + Veros ocean on real
bathymetry + slab land + Winton sea ice) has **no river routing**. Precipitation
minus evaporation over land (about 1.5–1.6 Sv globally in the control runs) is
not passed to the ocean at all; it simply leaves the system. The owner is
investigating a spurious North Pacific overturning (PMOC about 28–35 Sv,
observed about 0) and a North Pacific halocline bias, and missing river
freshwater is one suspect. This repository is meant to hold the routing map
(and later, possibly the code that applies it).

## What is here

- `make_river_routing.py`: builds a **nearest-coast** routing map. Copied
  unchanged from the case directory (`tools/make_river_routing.py`, commit
  7b5cb99 in port-to-new-JEM), then changed (2026-11-05) to write an ESMF
  weight file as its main output (see below).
- `grid_ESM_collocated`: symlink to `../port-to-new-JEM/grid_ESM_collocated/`
  (relative link; it breaks unless the clone sits next to port-to-new-JEM). The tool
  reads two files from it:
  - `ocn_topo_JCM_T31.nc`: `landsea_mask` (1 = land), `lat`, `lon` on
    (nj, ni) = (48, 96).
  - `JCM_T31.SCRIP.nc`: `grid_area` (steradians), centres and corners, used
    for cell areas, a layout check and the weight file's grid variables.

## The rule the tool implements

- A coastal cell is an ocean cell with a land cell among its 8 neighbours
  (longitude wraps; nothing beyond the polar rows).
- Every land cell drains to the coastal cell with the nearest centre by
  great-circle distance, if that distance is <= `--max-distance-km`
  (default 500).
- Land cells farther than that get `-1` (lost, a bookkeeping bin); ocean cells
  get `-2`.
- Main output: an **ESMF offline weight file** (same layout as
  `grid_ESM_collocated/weight_algo-conserve_JCM_T31_to_JCM_T31.nc`), read
  directly by `ESMFRegridder` in `jem/utils/esmf_regrid.py`. Decisions agreed
  with the owner:
  - Input is a **flux per unit area** (kg m^-2 s^-1); `S = area_src / area_dst`,
    so the area-weighted total is conserved (normalization "destarea").
  - **River input only**: one entry per routed land cell; ocean cells and lost
    cells have no entry. Output is zero except on receiving coastal cells; the
    caller adds it to the ocean's own freshwater flux.
  - **No clipping**: negative land P−E passes through; clipping is the
    caller's job.
  - Flat index `j * ni + i + 1` (1-based, longitude fastest), `grid_dims = [96, 48]`.
    `mask_a` = land, `mask_b` = ocean, `frac_a` = 1 on routed land cells,
    `frac_b` = 1 on receiving coastal cells.
- `--diagnostics NC` also writes the old index-map file: `routing_index` (flat
  `j * ni + i`, or -1/-2), `routing_j`, `routing_i`, `distance_km`,
  `coastal_mask`, `landsea_mask`, `cell_area` (m^2), `lat`, `lon`.
  `--plot PNG` draws a check map.

Run (conda env `jax` was used; it needs numpy, xarray, and matplotlib +
cartopy only for `--plot`):

    python make_river_routing.py grid_ESM_collocated/ocn_topo_JCM_T31.nc \
        grid_ESM_collocated/JCM_T31.SCRIP.nc weight_river_routing_JCM_T31.nc \
        --diagnostics river_routing_JCM_T31.nc --plot figures/river_routing.png

Use it in JEM like any weight file (fields are (n_lon, n_lat)):

    river = ESMFRegridder("weight_river_routing_JCM_T31.nc")(land_P_minus_E)

Verified 2026-11-05 with `ESMFRegridder`: the weights reproduce the index map
exactly at 500 and 1500 km; a uniform land flux comes out only on coastal
cells and conserves the routed total to ~1e-9 (the regridder stores weights in
float32). With the last 10 years of `winton_ctl_sst10` land P−E at 1500 km:
1.55 Sv total, 1.41 Sv routed; NW Pacific 0.116 Sv, NE Pacific 0.095 Sv, inside
the ranges in the table below.

Verified 2026-11-05: the index map (now the `--diagnostics` file) is identical to the map made in the case
directory. At 500 km, 667 of 1980 land cells are routed and 1313 (67.5% of
land area) are lost; at 3000 km essentially nothing is lost.

## Known limitations

- **Not real river basins.** Each land cell goes to its own nearest coast, so a
  big basin (Amazon, Amur, Mississippi, Yangtze, Siberian rivers) is split
  across several coasts instead of reaching its true mouth. Regional totals are
  fine; the coastal position of the input is not.
- Interior cells beyond the distance cap are lost; the 500 km default loses
  about half of land P−E. 1500 km loses about 0.13–0.15 Sv; 3000 km about 0.
- At T31, Hudson Bay and the Baltic are land and the Mediterranean is one cell,
  so their runoff lands on the open Atlantic coast.
- Negative land P−E (dry cells) would be routed too if fluxes are summed
  naively; decide whether to clip at zero.

## Results already obtained with it (last 10 years of the control runs)

Land P−E from the atmosphere output `surface_exchange.precipitation` −
`surface_exchange.evaporation` (kg m^-2 s^-1) in
`/p/tmp/tienyiao/output_T31/<run>/atm-*.nc`, runs `winton_ctl_sst10` (the
control) and `winton_ctl_woa`. At a 1500 km cap, freshwater that would reach:

| Destination | Model (Sv) | Real-world rough value (from memory, not checked) |
|---|---|---|
| NW Pacific, 20–66N, 110–180E | 0.11–0.14 | 0.08–0.10 |
| NE Pacific, 20–66N, 180–260E | 0.09–0.12 | 0.06–0.08 |
| Subpolar N Atlantic, 45–66N | 0.10 | 0.08–0.10 |
| Subtropical N Atlantic, 20–45N | 0.035–0.04 | 0.04–0.05 |
| Tropical N Atlantic, 0–20N (includes the Amazon's water) | 0.26 | 0.25–0.27 |
| Arctic | 0.10–0.11 | 0.10–0.13 |

So routing would add freshwater of realistic size to both northern basins,
and 3–10x the model's own subpolar North Pacific ocean P−E (0.01–0.035 Sv).

## Possible next steps (not started; confirm with the owner first)

1. Replace nearest-coast with basin-aware routing (e.g. a coarse D8 flow
   direction from `elevation` in `ocn_topo_JCM_T31.nc`, or a regridded
   observed river-mouth dataset).
2. Apply the map in JEM: add the routed land P−E to the ocean freshwater flux
   in the case's `model_setup.py` (the ocean forcing uses
   `seaice["derived"].ocean_freshwater_flux_up`, positive upward, so river
   input enters with a negative sign), spread over the destination cell or a
   small coastal neighbourhood, and check the global freshwater budget closes.
3. Rerun the control and compare PMOC and North Pacific salinity.

## Conventions the owner wants

- Commit only when asked; commit messages end with the attribution lines the
  harness provides.
- Explain results plainly with numbers; keep figures in a `figures/` folder.
