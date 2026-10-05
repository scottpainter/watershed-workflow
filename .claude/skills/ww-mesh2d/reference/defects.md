# Defect kinds, proposals, and worked examples

`diagnostics.findDefects()` reports only what Watershed Workflow will not fix itself.
`repair.proposeFixes()` offers ranked `Proposal`s (action + JSON params + preview + plot
geometry + caveats). Actions: `dropEmptyReaches`, `dropReaches`, `offsetDivide`, `splitHUC`,
`moveRiverNode`, `sequence` (several actions as one recipe step).

| kind | stage | meaning | proposals |
|---|---|---|---|
| `empty-geometry` | rivers | reaches with empty/missing geometry; `createRivers` fails on them | drop them, rewiring hydroseq links around them |
| `hydroseq-vs-geometry` | rivers | downstream links by hydroseq differ from the geometry | none -- usually a symptom (empty rows, braids); rescan after other fixes |
| `createRivers` / `watershed` | rivers / hucs | the network or the HUC polygons cannot be built at all; `watershed` also names two HUCs sharing 2+ separate stretches of boundary (WW cannot split those) | none -- inspect |
| `multiple-crossing` | simplify | a reach crosses one HUC boundary several times (stream digitized along a divide); cut-and-snap fails | `offsetDivide`: move the divide off the reach (and other reaches running along it) by the clearance; the HUC holding most of the reach keeps it; reverse direction offered second |
| `exterior-crossing`, `continuity`, `snapReachEndpoints` | simplify | other cut-and-snap failures | none -- inspect |
| `sharp-angles` | simplify | `smoothSharpAngles` failed | none -- often caused by a nearby defect; rescan after fixing it |
| `multiple-outlets` | outlets | the network leaves a HUC at 2+ places | (1) drop a separate network leaving on its own (ranked first if its reachcodes are from another HUC8); (2) outlets into the same HUC < 2 x reach length apart: snap their confluence onto the divide, then a flat-ended offset (`sequence`), round-ended offset second; (3) split the HUC along a cut separating the extra outlet's network: shortest straight cuts across a neck, or the line midway between the networks, each as a new HUC or merged into the HUC it drains to |
| `corridor-overlap` | mesh | river corridors overlap | none -- inspect (reach spacing vs widths) |
| `small-cells` | mesh | cells below `min_cell_area`, grouped into sites | none -- see SKILL.md step 4 |

The split-cut generators are purely geometric (ignore topography; single cut only). A more
robust method (NHDPlus catchment polygons via `featureid`, DEM delineation, multi-part cuts)
is planned -- say so when offering a cut.

## Worked examples (Tiffin River, HUC8 04100006, 26 HUC12s)

All approved one at a time by the user; the recipe replays them exactly.

- **Empty rows (179)** in a colleague's export: `mergeShortReaches` blanks merged reaches and
  only `resetDataFrame()` drops them; the file was saved in between. From USGS sources the
  defect does not occur. Fix when present: drop, rewiring hydroseq.
- **Bean Creek along the 0202|0203 divide**, crossing it 4-8 times: `offsetDivide` into 0203,
  61.6 m (2.5 x order-4 width) off three reaches, 0.20 km2 moved. Rejected: moving it into 0202
  (more area, and 0203 holds less of the reach).
- **0202: two outlets 38 m apart into 0205.** The confluence of Bean Creek and 15661626 lay 9 m
  inside a narrow tongue of 0202; `simplify()` snapped it onto the divide, but Bean Creek
  wiggled across the tongue's edge 38 m upstream, leaving a stub in 0205 (and a
  `smoothSharpAngles` failure). The user chose "snap the confluence to the boundary first, then
  the offset": `moveRiverNode` (9.2 m) + flat-ended `offsetDivide` into 0205 (0.036 km2), so the
  confluence is 0202's single outlet. The round-ended offset (confluence interior, outlet a
  crossing of the downstream reach) was the alternative.
- **0104: two outlets** -- its eastern arm drains via Garrison Drain into 0106 3.4 km from the
  main outlet. Split along a 640 m straight cut across the neck into a new HUC 041000060104a
  (tohuc 0106). Rejected: merging the arm into 0106 (0104 -31 %, odd shape); deleting the drain.
- **0604: second outlet to the domain exterior** from 15667491, a 1-reach headwater of the
  neighbouring HUC8 (reachcode 04100005) clipped by the domain: dropped.
- **Small cells** (pass 2): HUC boundaries running beside corridors gave cells down to 0.06 m2;
  fixed in WW by snapping interior HUC boundaries to corridors (15 m) and corridor vertices to
  the exterior boundary (5 m). Final mesh 304,478 cells, smallest 81.0 m2.
