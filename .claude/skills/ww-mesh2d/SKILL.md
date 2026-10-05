---
name: ww-mesh2d
description: Prepare HUC boundaries and an NHDPlus river network with Watershed Workflow and build a river-aligned 2D surface mesh for ATS. Covers downloading WBD HUCs and NHDPlus flowlines, pruning the network by contributing area, finding the topological defects WW will not fix itself (multiple outlets per HUC, streams along divides, ...), proposing fixes for user approval one at a time, and meshing with a minimum-cell-area check. Use it also when the user ONLY wants repaired HUC/river data to continue in their own Jupyter notebook, or starts from geopackages exported partway through a WW workflow.
---

# Watershed Workflow: repaired hydrography and the 2D mesh

The goal is a river-aligned 2D mesh (quads along stream corridors, triangles elsewhere) whose
HUC boundaries and river network are topologically consistent. **The hydrography must be
repaired before meshing** -- the standard WW notebook workflow (`simplify()` ->
`tessalateRiverAligned()`) fails or silently produces a wrong `tohuc` on raw data.

## 0. Establish the goal and entry point first

Ask (AskUserQuestion) unless the user already said:

1. **Goal**
   - **Repair only** -- corrected HUC + river geopackages (and a replayable recipe) that the
     user loads into their own Jupyter notebook. Many users work in notebooks; for them, stop
     after step 3 and hand off (see `reference/notebook.md`). Do not build the mesh.
   - **Repair + 2D mesh** -- continue through step 4.
2. **Entry point**
   - **From USGS sources**: a HUC code (domain) and the HUC level of the sub-watersheds.
   - **Their own geopackages** (HUCs + rivers). Often exported from a notebook partway through
     a workflow -- check them (step 1b).
3. **Pruning threshold** (from-source only): an ABSOLUTE contributing area in km2. Do not use a
   fraction of the domain area (the Coweeta example's `prune_by_area_fraction`) -- users found
   it far too aggressive. Ask; 2 km2 was used for the 2014 km2 Tiffin basin.

## Requirements

- Watershed Workflow with `watershed_workflow.diagnostics` and `watershed_workflow.repair`
  (branch `tohuc-single-outlet-and-split-hydroseq` of github.com/scottpainter/watershed-workflow
  until merged). Check: `python -c "import watershed_workflow.diagnostics, watershed_workflow.repair"`.
  Use the conda env WW is installed in (find it; don't assume `python` on PATH).
- Run everything from a working directory for the basin (HyRiver caches downloads in `./cache`).
  Set `MPLBACKEND=Agg`: WW calls `plt.show()` inside some functions, which blocks on macOS.
- Scripts are in this skill's `scripts/` directory; each has `--help`.

## 1a. From USGS sources

```bash
python scripts/fetch_and_prune.py --huc <HUC> --level 12 --prune-km2 <A> --out .
```
WBD HUCs + NHDPlus MR v2.1 flowlines (with catchments and bankfull widths) ->
`createRivers(method='hydroseq')` -> `reduceRivers(prune_by_area=A, remove_braided_divergences=True)`.
Braids are removed by default: kept, they create false second outlets and hydroseq/geometry
mismatches. Diversions are kept unless `--remove-diversions`. Writes `hucs_raw.gpkg`,
`rivers_pruned.gpkg` and an empty `recipe.json` recording these settings.
Copy them to the working state: `cp hucs_raw.gpkg state_hucs.gpkg; cp rivers_pruned.gpkg state_rivers.gpkg`.

## 1b. From the user's geopackages -- check them first

- **Empty geometries + a `do-not-merge` column** => exported mid-`simplify()` (after
  `mergeShortReaches`, before `river.resetDataFrame()`). The empty rows are an export artifact;
  the scan reports them (`empty-geometry`) and proposes dropping them with hydroseq rewiring.
  Tell the user why they appeared.
- **A `new_preorder_index` column** => the river dataframe was written with its WW-internal
  index; `createRivers()` will refuse it. Drop the column.
- CRS projected in metres (EPSG:5070 for CONUS); HUCs have an ID column and `tohuc`; rivers
  have `ID`, `hydroseq`/`dnhydroseq`/`uphydroseq`, `stream_order`, `drainage_area_sqkm`,
  `bankfull_width` (needed for corridor widths and clearances).
- Start a recipe for them: `python -c "import watershed_workflow.repair as r; r.Recipe(metadata=dict(hucs='...', rivers='...')).save('recipe.json')"`.

## 2. Repair loop -- one defect at a time, nothing applied without approval

```bash
python scripts/scan.py --hucs state_hucs.gpkg --rivers state_rivers.gpkg --tag s0 --recipe recipe.json
```
`findDefects()` runs the WW pipeline and reports ONLY what WW will not fix itself; things WW
handles (e.g. `tohuc` updates) are notes. `proposeFixes()` lists ranked fixes per defect.
Then, for **each defect, in order**:

1. Show the geometry: `python scripts/plot_defect.py --scan s0 --defect i --hucs ... --rivers ...`
   (overview + zoom, proposals overlaid). Look at the figure yourself before showing it; HUC
   labels come from the data -- never place labels by hand.
2. Explain the defect in plain terms, list the proposals with their previews and caveats,
   recommend one, and **wait for the user's decision**. The user may approve, reject, ask for a
   variant (edit `params`), or propose something else.
3. Apply only the approved one: `python scripts/apply_step.py --scan s0 --defect i --proposal j
   --hucs state_hucs.gpkg --rivers state_rivers.gpkg --recipe recipe.json --by <user> --note "..."`.
   Record rejections too (`--reject`) so they are not offered again.
4. **Rescan** (new tag) before the next defect -- fixes interact (one can remove, move or
   create others; e.g. a `sharp-angles` failure is often a side effect of a nearby defect).

When a defect has **no proposal**, investigate before designing anything: run WW's own
`simplify()` on the area and see what it does with it (WW handles many cases -- confluences
near triple junctions, outlet tips -- natively). Then show the geometry and offer options.
See `reference/defects.md` for each defect kind, its proposals and worked examples.

Finish when the scan reports 0 defects. Check the recipe replays:
`repair.applyRecipe(hucs_raw, rivers_pruned, 'recipe.json')` must reproduce the state exactly.

## 3. Hand off (repair-only users stop here)

Write `<name>_hucs_corrected.gpkg` and `<name>_rivers_corrected.gpkg` (copies of the state) and
keep `recipe.json` beside them. Give the user the notebook cells in `reference/notebook.md`
(load the corrected data, check it, continue with `simplify()` / `tessalateRiverAligned()`),
including the settings the scan used -- the repairs were verified with those.

## 4. Build the 2D mesh

```bash
python scripts/build_mesh2d.py --hucs state_hucs.gpkg --rivers state_rivers.gpkg --out mesh [--elevate]
```
`simplify(reach 60 m, HUC 150 m, snap_triple_junctions_tol 50 m)` + `tessalateRiverAligned`
(corridor width = mean bankfull width per stream order, `refine_min_angle=32`; WW snaps HUC
boundaries to corridors and corridors to the domain boundary by default). Reports cells by
type and the smallest cells.

**Mesh quality criterion: no cell smaller than 50 m2** (ask if the user has another). Small
cells limit the ATS time step; skinny cells that are not small are fine. If cells fall below
it, run `scan.py --mesh --min-cell-area 50` for `small-cells` sites, plot them, and discuss
(typically a HUC boundary running just beside a river corridor; fixes so far: larger
`snap_hucs_to_corridors_tol`, or `offsetDivideFromReaches` off that reach).

`--elevate` adds the 3DEP 10 m DEM (py3dep static tiles; WW's `Manager3DEP` uses the dynamic
map service, which times out on large domains), river profiles and burn-in, pit filling, and
HUC/outlet/corridor/stream-order regions -- the rest of the ATS-ready 2D mesh. Timing on the
2014 km2 Tiffin basin (304,478 cells): planform 4.5 min; `--elevate` about 1 hour more -- run it
in the background and tell the user.

## Conventions

- Confluence terms: **main/side branch** by NHD `uphydroseq` only; **outlet branch** comes from
  the HUC whose outlet this is; **through branch** from that HUC's `tohuc` HUC; **downstream
  reach** leaves the confluence. Never say "receiving stream".
- Clearance of a divide from a reach = 2.5 x the mean bankfull width of the reach's stream order.
- Every HUC must have exactly one outlet; WW (`updateToHUCs`) derives `tohuc` from the geometry
  and raises `MultipleOutletsError` otherwise -- the scan reports these as `multiple-outlets`.
- Report numbers from the data (IDs, distances, areas); say when something was not verified.

## Pitfalls

- `Watershed` and `River` objects do not pickle (a lambda inside); checkpoint GeoDataFrames
  (`hucs.to_dataframe()`) and the `Mesh2D`, not them.
- After `tessalateRiverAligned()` the `Watershed`'s polygons cannot be rebuilt (HUC linestrings
  were adjusted in place) -- save `hucs.deepcopy()` before meshing if you need polygons later.
- Plot meshed HUC boundaries, not pre-mesh ones, when judging conformance of the mesh.
- `reduceRivers(prune_by_area=...)` is absolute km2 despite examples converting from a fraction.
