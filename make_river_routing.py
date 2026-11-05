"""Build a nearest-coast river routing map for a land-sea mask.

Rule:
  * A coastal cell is an ocean cell with at least one land cell among its eight
    neighbours (longitude wraps around; the first and last latitude rows have
    no neighbours beyond them).
  * Every land cell is routed to the coastal cell whose centre is nearest by
    great-circle distance, if that distance is at most `--max-distance-km`
    (default 500 km).
  * A land cell farther than that from any coastal cell is routed to -1, a
    bookkeeping bin: the freshwater it produces leaves the system and is
    counted as lost rather than put into the ocean.
  * Ocean cells are not routed and hold -2.

Main output: an ESMF offline weight file (same layout as ESMF_RegridWeightGen
output, readable by jem/utils/esmf_regrid.py's ESMFRegridder), source and
destination both the same grid:
  row, col, S     sparse matrix (1-based flat indices, longitude fastest, i.e.
                  j * ni + i + 1): one entry per routed land cell, col = the
                  land cell, row = its coastal cell,
                  S = area(land cell) / area(coastal cell)
  The input field must be a flux per unit area (e.g. kg m-2 s-1). The output
  is the river input per unit area of the destination cells, zero elsewhere;
  area-weighted totals are conserved apart from the lost cells, which have no
  entry. Ocean cells have no entry either: the output is river input only and
  must be added to the ocean's own freshwater flux by the caller. Negative
  values (land P-E < 0) pass through unchanged; clipping is up to the caller.
  src_grid_dims, dst_grid_dims   (ni, nj), as in ESMF files
  xc/yc, xv/yv, area (square radians)   from the SCRIP grid, _a and _b
  mask_a          1 on land (cells that may be routed)
  mask_b          1 on ocean
  frac_a          1 on routed land cells, 0 elsewhere (0 on lost cells)
  frac_b          1 on coastal cells that receive water, 0 elsewhere

Optional --diagnostics file (netCDF, on the mask file's (nj, ni) layout):
  routing_index   flat index j * ni + i of the destination coastal cell, or
                  -1 (lost) / -2 (ocean cell)
  routing_j, routing_i   the same destination as (nj, ni) indices, or -1/-2
  distance_km     distance to the destination (to the nearest coastal cell for
                  a lost cell), NaN over ocean
  coastal_mask    1 on coastal ocean cells
  landsea_mask    copied from the input (1 = land)
  cell_area       m^2, from the SCRIP grid (for converting fluxes per unit
                  area into volume fluxes)
  lat, lon        cell centres

Example (from this repository's top directory):
    python make_river_routing.py grid_ESM_collocated/ocn_topo_JCM_T31.nc \\
        grid_ESM_collocated/JCM_T31.SCRIP.nc weight_river_routing_JCM_T31.nc \\
        --diagnostics river_routing_JCM_T31.nc --plot figures/river_routing.png
"""
import argparse

import numpy as np
import xarray as xr

R_EARTH_KM = 6371.0
OCEAN, LOST = -2, -1

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("mask_file", help="Land-sea mask file with landsea_mask (1 = land) and lat/lon on (nj, ni).")
parser.add_argument("scrip_file", help="SCRIP grid file of the same grid, for the cell areas.")
parser.add_argument("out_file", help="Output ESMF weight file (netCDF).")
parser.add_argument("--diagnostics", metavar="NC",
                    help="Also write the routing as (nj, ni) index maps, distances and masks, for checking.")
parser.add_argument("--max-distance-km", type=float, default=500.0,
                    help="Farthest a land cell may be from its coastal cell (default 500).")
parser.add_argument("--plot", metavar="PNG", help="Also draw a map of the routing for checking.")
args = parser.parse_args()

mask_ds = xr.open_dataset(args.mask_file)
land = mask_ds["landsea_mask"].values == 1                      # (nj, ni)
lat = mask_ds["lat"].values
lon = mask_ds["lon"].values
nj, ni = land.shape
scrip = xr.open_dataset(args.scrip_file)
area = scrip["grid_area"].values.reshape(nj, ni) * (R_EARTH_KM * 1e3) ** 2   # steradians -> m^2
if not np.allclose(scrip["grid_center_lat"].values.reshape(nj, ni), lat):
    raise SystemExit("The SCRIP grid's cell centres do not match the mask file's (nj, ni) layout.")

# coastal ocean cells: an ocean cell with a land cell among its eight neighbours
padded = np.pad(land, ((1, 1), (0, 0)), constant_values=False)    # no neighbours beyond the polar rows
land_neighbour = np.zeros_like(land)
for dj in (-1, 0, 1):
    for di in (-1, 0, 1):
        if dj or di:
            land_neighbour |= np.roll(padded, di, axis=1)[1 + dj:1 + dj + nj]
coastal = ~land & land_neighbour


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine distance (km) between every point of set 1 and every point of set 2."""
    p1, p2 = np.deg2rad(lat1)[:, None], np.deg2rad(lat2)[None, :]
    dl = np.deg2rad(lon2)[None, :] - np.deg2rad(lon1)[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


land_j, land_i = np.nonzero(land)
coast_j, coast_i = np.nonzero(coastal)
dist = great_circle_km(lat[land_j, land_i], lon[land_j, land_i], lat[coast_j, coast_i], lon[coast_j, coast_i])
nearest = np.argmin(dist, axis=1)
nearest_km = dist[np.arange(len(land_j)), nearest]
routed = nearest_km <= args.max_distance_km

routing_j = np.full((nj, ni), OCEAN, dtype=np.int32)
routing_i = np.full((nj, ni), OCEAN, dtype=np.int32)
routing_j[land_j, land_i] = np.where(routed, coast_j[nearest], LOST)
routing_i[land_j, land_i] = np.where(routed, coast_i[nearest], LOST)
routing_index = np.where(routing_j >= 0, routing_j * ni + routing_i, routing_j).astype(np.int32)
distance_km = np.full((nj, ni), np.nan)
distance_km[land_j, land_i] = nearest_km

dims = ("nj", "ni")
out = xr.Dataset(
    {
        "routing_index": (dims, routing_index, {
            "long_name": "flat index j * ni + i of the coastal ocean cell a land cell drains to",
            "flag_values": np.array([LOST, OCEAN], dtype=np.int32),
            "flag_meanings": "lost_beyond_max_distance ocean_cell_not_routed"}),
        "routing_j": (dims, routing_j, {"long_name": "nj index of the destination coastal cell (-1 lost, -2 ocean)"}),
        "routing_i": (dims, routing_i, {"long_name": "ni index of the destination coastal cell (-1 lost, -2 ocean)"}),
        "distance_km": (dims, distance_km, {"units": "km",
                                             "long_name": "great-circle distance from a land cell to the nearest coastal ocean cell"}),
        "coastal_mask": (dims, coastal.astype(np.int32), {"long_name": "1 on ocean cells with a land neighbour (8-connected)"}),
        "landsea_mask": (dims, land.astype(np.int32), {"long_name": "1 = land, 0 = ocean (copied from the input)"}),
        "cell_area": (dims, area, {"units": "m2", "long_name": "cell area from the SCRIP grid"}),
    },
    coords={"lat": (dims, lat), "lon": (dims, lon)},
    attrs={
        "title": "Nearest-coast river routing",
        "rule": "land cell -> nearest coastal ocean cell (great-circle, cell centres) if within max_distance_km, "
                "else -1 (freshwater lost, for bookkeeping); ocean cells -2",
        "max_distance_km": args.max_distance_km,
        "mask_file": args.mask_file,
        "scrip_file": args.scrip_file,
    },
)
if args.diagnostics:
    out.to_netcdf(args.diagnostics)

# ESMF weight file. Flat index j * ni + i is the SCRIP/ESMF order (grid_dims = [ni, nj], longitude fastest);
# the layout check against grid_center_lat above guarantees the SCRIP cells are in this order.
area_sr = scrip["grid_area"].values                                   # steradians, flat
src = land_j[routed] * ni + land_i[routed]
dst = coast_j[nearest[routed]] * ni + coast_i[nearest[routed]]
order = np.argsort(dst, kind="stable")                                # sorted by row, like ESMF output
src, dst = src[order], dst[order]
S = area_sr[src] / area_sr[dst]
frac_a = np.zeros(nj * ni); frac_a[src] = 1.0
frac_b = np.zeros(nj * ni); frac_b[dst] = 1.0
grid_dims = np.array([ni, nj], dtype=np.int32)
center = {"yc": scrip["grid_center_lat"].values, "xc": scrip["grid_center_lon"].values}
corner = {"yv": scrip["grid_corner_lat"].values, "xv": scrip["grid_corner_lon"].values}
deg = {"units": "degrees"}
wvars = {
    "src_grid_dims": ("src_grid_rank", grid_dims),
    "dst_grid_dims": ("dst_grid_rank", grid_dims),
    "col": ("n_s", (src + 1).astype(np.int32)),
    "row": ("n_s", (dst + 1).astype(np.int32)),
    "S": ("n_s", S),
}
for side, mask, frac in (("a", land, frac_a), ("b", ~land, frac_b)):
    n, nv = f"n_{side}", f"nv_{side}"
    for name, v in center.items():
        wvars[f"{name}_{side}"] = (n, v, deg)
    for name, v in corner.items():
        wvars[f"{name}_{side}"] = ((n, nv), v, deg)
    wvars[f"mask_{side}"] = (n, mask.ravel().astype(np.int32), {"units": "unitless"})
    wvars[f"area_{side}"] = (n, area_sr, {"units": "square radians"})
    wvars[f"frac_{side}"] = (n, frac, {"units": "unitless"})
weights = xr.Dataset(wvars, attrs={
    "title": "Nearest-coast river routing weights (ESMF offline weight file layout)",
    "normalization": "destarea",
    "map_method": "Nearest-coast river routing",
    "rule": "land cell -> nearest coastal ocean cell (great-circle, cell centres) if within max_distance_km; "
            "S = area_src / area_dst, so a per-unit-area flux is conserved; lost land cells and ocean cells have "
            "no entry (output is river input only)",
    "max_distance_km": args.max_distance_km,
    "conventions": "NCAR-CSM",
    "domain_a": args.scrip_file, "domain_b": args.scrip_file,
    "grid_file_src": args.scrip_file, "grid_file_dst": args.scrip_file,
    "mask_file": args.mask_file,
})
weights.to_netcdf(args.out_file, encoding={v: {"_FillValue": None} for v in weights.data_vars})

n_land, n_lost = len(land_j), int(np.sum(~routed))
lost_area = area[land_j[~routed], land_i[~routed]].sum()
print(f"{n_land} land cells: {n_land - n_lost} routed to {len(np.unique(nearest[routed]))} of {coastal.sum()} "
      f"coastal cells, {n_lost} lost (> {args.max_distance_km:g} km; {lost_area / area[land].sum():.1%} of the land area)")
print(f"routed distance: median {np.median(nearest_km[routed]):.0f} km, max {nearest_km[routed].max():.0f} km")
print(f"saved {args.out_file} ({len(S)} weights)")
if args.diagnostics:
    print("saved", args.diagnostics)

if args.plot:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_subplot(projection=ccrs.Robinson(central_longitude=180))
    ax.set_global()
    shown = np.where(land, distance_km, np.nan)
    p = ax.scatter(lon[land], lat[land], c=shown[land], s=14, marker="s", cmap="YlOrBr",
                   vmin=0, vmax=args.max_distance_km, transform=pc, zorder=2)
    lost_cells = land & (routing_index == LOST)
    ax.scatter(lon[lost_cells], lat[lost_cells], s=14, marker="s", color="k", transform=pc, zorder=3,
               label=f"lost (> {args.max_distance_km:g} km): {n_lost} cells")
    ax.scatter(lon[coastal], lat[coastal], s=6, marker="o", color="tab:blue", transform=pc, zorder=2,
               label="coastal ocean cell")
    # a line from every routed land cell to its destination; Geodetic draws the great circle
    for j, i in zip(*np.nonzero(land & (routing_index >= 0))):
        jj, ii = routing_j[j, i], routing_i[j, i]
        ax.plot([lon[j, i], lon[jj, ii]], [lat[j, i], lat[jj, ii]], color="0.35", lw=0.4,
                transform=ccrs.Geodetic(), zorder=1)
    ax.coastlines(lw=0.4, color="red")
    plt.colorbar(p, ax=ax, shrink=0.6, label="distance to the destination coastal cell (km)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Nearest-coast river routing, max distance {args.max_distance_km:g} km "
                 f"(grey lines: land cell -> coastal ocean cell)", fontsize=11)
    import os
    os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
    fig.savefig(args.plot, dpi=110, bbox_inches="tight")
    print("saved", args.plot)
