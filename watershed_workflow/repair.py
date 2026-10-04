"""Propose, review and replay fixes for the defects found by diagnostics.findDefects().

Nothing here changes data on its own.  The intended loop is:

1. report = diagnostics.findDefects(hucs, rivers, ...)
2. proposals = repair.proposeFixes(report, hucs, rivers)
3. for one defect, show its proposals to a person, who approves one, rejects
   them, or edits a proposal's params;
4. hucs, rivers = proposal.apply(hucs, rivers); recipe.record(proposal, 'approved')
5. rescan (fixes interact: one may remove, move or create other defects)
   and repeat until the scan is clean.

The Recipe is the record of those decisions.  Each step stores an action
name and JSON-serializable params, so the corrected data can be rebuilt
from the original inputs with applyRecipe(), and a reviewer can read every
decision in one place.

Actions are pure functions (hucs, rivers, **params) -> (hucs, rivers) that
return new GeoDataFrames; they are registered in ACTIONS.  Proposers map a
Defect to a list of ranked Proposals; they are registered in PROPOSERS by
defect kind.  A defect kind with no proposer (or for which no proposal
applies) gets an empty list and has to be fixed by hand.

Example
-------
>>> report = watershed_workflow.diagnostics.findDefects(hucs, rivers, 60., 150., mesh=False)
>>> proposals = watershed_workflow.repair.proposeFixes(report, hucs, rivers)
>>> for defect, options in zip(report.defects, proposals):
...     print(defect.message)
...     for p in options:
...         print('  ', p.description, p.preview)
>>> recipe = watershed_workflow.repair.Recipe()
>>> hucs, rivers = proposals[0][0].apply(hucs, rivers)
>>> recipe.record(proposals[0][0], 'approved', by='me')
>>> recipe.save('recipe.json')
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import copy
import dataclasses
import datetime
import json
import logging
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import shapely.geometry
import shapely.ops
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

import watershed_workflow.hydro.river
import watershed_workflow.hydro.watershed
import watershed_workflow.mesh.river_mesh
import watershed_workflow.sources.standard_names as names
from watershed_workflow.diagnostics import Defect, DefectReport

__all__ = ['Proposal', 'Recipe', 'proposeFixes', 'applyRecipe', 'ACTIONS', 'PROPOSERS',
           'dropEmptyReaches', 'dropReaches', 'offsetDivide', 'splitHUC',
           'straightNeckCuts', 'equidistantCut']

Frames = Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]


#
# Actions: pure (hucs, rivers, **params) -> (hucs, rivers)
#
def dropEmptyReaches(hucs : gpd.GeoDataFrame,
                     rivers : gpd.GeoDataFrame,
                     hydroseq_col : str = names.HYDROSEQ,
                     dn_col : str = names.DOWNSTREAM_HYDROSEQ,
                     up_col : str = names.UPSTREAM_HYDROSEQ,
                     area_col : str = names.DRAINAGE_AREA) -> Frames:
    """Drop reaches with missing or empty geometry, rewiring hydroseq links around them.

    A live reach whose downstream link names an empty row would be cut off
    from the network when rivers are built by hydroseq.  So each live
    reach's downstream link is advanced through any chain of empty rows to
    the first live reach (or to a value not in the table, i.e. an outlet),
    and an upstream link naming an empty row is replaced by the
    largest-drainage live reach now draining into it, else 0.  The links are
    only rewired if the hydroseq columns are present.
    """
    empty = rivers.geometry.isna() | rivers.geometry.is_empty
    out = rivers[~empty].copy()

    if all(c in rivers for c in (hydroseq_col, dn_col, up_col)):
        dead = rivers[empty]
        dead_dn = dict(zip(dead[hydroseq_col], dead[dn_col]))

        def resolveDown(hs):
            seen = set()
            while hs in dead_dn and hs not in seen:
                seen.add(hs)
                hs = dead_dn[hs]
            return hs
        out[dn_col] = [resolveDown(hs) for hs in out[dn_col]]

        dead_hs = set(dead[hydroseq_col])
        for i in out.index[out[up_col].isin(dead_hs)]:
            feeders = out[out[dn_col] == out.at[i, hydroseq_col]]
            if len(feeders) == 0:
                out.at[i, up_col] = 0.
            elif area_col in feeders:
                out.at[i, up_col] = feeders.sort_values(area_col).iloc[-1][hydroseq_col]
            else:
                out.at[i, up_col] = feeders[hydroseq_col].iloc[0]

    logging.info(f'dropEmptyReaches: dropped {int(empty.sum())} reaches')
    return hucs, out.reset_index(drop=True)


def dropReaches(hucs : gpd.GeoDataFrame,
                rivers : gpd.GeoDataFrame,
                reaches : List[Any],
                id_col : str = names.ID) -> Frames:
    """Drop the reaches with these IDs."""
    drop = rivers[id_col].astype(str).isin([str(r) for r in reaches])
    missing = set(str(r) for r in reaches) - set(rivers.loc[drop, id_col].astype(str))
    if missing:
        raise ValueError(f'dropReaches: no reaches with ID {sorted(missing)}')
    return hucs, rivers[~drop].reset_index(drop=True)


def offsetDivide(hucs : gpd.GeoDataFrame,
                 rivers : gpd.GeoDataFrame,
                 keep : Any,
                 other : Any,
                 reaches : List[Any],
                 clearance : Dict[str, float],
                 id_col : str = names.ID,
                 huc_id_col : str = names.ID) -> Frames:
    """hydro.watershed.offsetDivideFromReaches() with a per-reach clearance table.

    clearance maps each reach ID (as a string) to its clearance, so the step
    replays exactly however the rivers change before it.
    """
    sel = rivers[rivers[id_col].astype(str).isin([str(r) for r in reaches])]
    if len(sel) != len(reaches):
        raise ValueError(f'offsetDivide: reaches {reaches} not all found')
    hucs, _ = watershed_workflow.hydro.watershed.offsetDivideFromReaches(
        hucs, sel, keep, other, lambda row: clearance[str(row[id_col])], id_col=huc_id_col)
    return hucs, rivers


def splitHUC(hucs : gpd.GeoDataFrame,
             rivers : gpd.GeoDataFrame,
             **params) -> Frames:
    """hydro.watershed.splitHUC(): split a HUC along a cut into a new HUC, or into a neighbor."""
    hucs, _ = watershed_workflow.hydro.watershed.splitHUC(hucs, **params)
    return hucs, rivers


ACTIONS : Dict[str, Callable[..., Frames]] = {
    'dropEmptyReaches' : dropEmptyReaches,
    'dropReaches' : dropReaches,
    'offsetDivide' : offsetDivide,
    'splitHUC' : splitHUC,
}


#
# Proposals and recipes
#
def _defectRecord(defect : Optional[Defect]) -> Optional[dict]:
    if defect is None:
        return None
    rec = dict(stage=defect.stage, kind=defect.kind, message=defect.message,
               reaches=[str(r) for r in defect.reaches], hucs=[str(h) for h in defect.hucs])
    if defect.location is not None:
        rec['location'] = [round(defect.location.x, 3), round(defect.location.y, 3)]
    return rec


@dataclasses.dataclass
class Proposal:
    """A candidate fix for one defect.

    action : str
        Key of ACTIONS.
    params : dict
        Keyword arguments of the action; JSON-serializable, so the step can
        be recorded and replayed.  May be edited before applying.
    description : str
        What it does, in words.
    defect : Defect
        The defect it addresses.
    preview : dict
        What it would change (e.g. area moved, reaches dropped).
    geometry : gpd.GeoDataFrame, optional
        Shapes to plot when reviewing it (column 'role' says what each is).
    caveats : list of str
        Reasons a reviewer might not want it.
    """
    action : str
    params : Dict[str, Any]
    description : str
    defect : Optional[Defect] = None
    preview : Dict[str, Any] = dataclasses.field(default_factory=dict)
    geometry : Optional[gpd.GeoDataFrame] = None
    caveats : List[str] = dataclasses.field(default_factory=list)

    def apply(self, hucs : gpd.GeoDataFrame, rivers : gpd.GeoDataFrame) -> Frames:
        """Apply to copies of hucs and rivers, returning the new ones."""
        return ACTIONS[self.action](hucs.copy(), rivers.copy(), **self.params)

    def key(self) -> str:
        return json.dumps(dict(action=self.action, params=self.params), sort_keys=True, default=str)


class Recipe:
    """An ordered record of reviewed proposals, replayable on the original inputs.

    Each step is a dict: action, params, description, decision ('approved'
    or 'rejected'), by, date, note, and the defect it addressed.  Only
    approved steps are applied; rejected ones are kept so the same proposal
    is not offered again (see isRejected) and so the record is complete.
    """
    VERSION = 1

    def __init__(self, steps : Optional[List[dict]] = None, metadata : Optional[dict] = None):
        self.steps = list(steps) if steps is not None else []
        self.metadata = dict(metadata) if metadata is not None else {}

    def record(self, proposal : Proposal, decision : str, by : Optional[str] = None,
               note : Optional[str] = None) -> dict:
        if decision not in ('approved', 'rejected'):
            raise ValueError("decision must be 'approved' or 'rejected'")
        json.dumps(proposal.params)    # fail now, not on save
        step = dict(action=proposal.action, params=copy.deepcopy(proposal.params),
                    description=proposal.description, decision=decision, by=by,
                    date=datetime.date.today().isoformat(), note=note,
                    defect=_defectRecord(proposal.defect))
        self.steps.append(step)
        return step

    @property
    def approved(self) -> List[dict]:
        return [s for s in self.steps if s['decision'] == 'approved']

    def isRejected(self, proposal : Proposal) -> bool:
        return any(s['decision'] == 'rejected'
                   and Proposal(s['action'], s['params'], '').key() == proposal.key() for s in self.steps)

    def apply(self, hucs : gpd.GeoDataFrame, rivers : gpd.GeoDataFrame) -> Frames:
        """Apply the approved steps, in order, to copies of hucs and rivers."""
        for i, s in enumerate(self.approved):
            logging.info(f'recipe step {i}: {s["action"]} -- {s["description"]}')
            hucs, rivers = ACTIONS[s['action']](hucs.copy(), rivers.copy(), **s['params'])
        return hucs, rivers

    def to_dict(self) -> dict:
        return dict(watershed_workflow_recipe=self.VERSION, metadata=self.metadata, steps=self.steps)

    def save(self, filename : str) -> None:
        with open(filename, 'w') as fid:
            json.dump(self.to_dict(), fid, indent=2)

    @classmethod
    def load(cls, filename : str) -> 'Recipe':
        with open(filename) as fid:
            d = json.load(fid)
        if d.get('watershed_workflow_recipe') != cls.VERSION:
            raise ValueError(f'{filename}: not a version {cls.VERSION} recipe')
        return cls(d['steps'], d.get('metadata'))


def applyRecipe(hucs : gpd.GeoDataFrame, rivers : gpd.GeoDataFrame, recipe : Recipe | str) -> Frames:
    """Rebuild corrected HUCs and rivers from the original inputs and a recipe (or its filename)."""
    if isinstance(recipe, str):
        recipe = Recipe.load(recipe)
    return recipe.apply(hucs, rivers)


#
# Proposers: (defect, hucs, rivers, options) -> List[Proposal], best first
#
def _live(rivers : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return rivers[~(rivers.geometry.isna() | rivers.geometry.is_empty)]


def _findReach(rivers : gpd.GeoDataFrame, reach_id : Any, id_col : str) -> Optional[str]:
    """The input ID of a reach named in a defect; simplify() may have split it (suffix a/b)."""
    ids = set(rivers[id_col].astype(str))
    rid = str(reach_id)
    while rid and rid not in ids and re.search('[a-z]$', rid):
        rid = rid[:-1]
    return rid if rid in ids else None


def _proposeDropEmpty(defect, hucs, rivers, opts) -> List[Proposal]:
    empty = rivers.geometry.isna() | rivers.geometry.is_empty
    has_links = all(c in rivers for c in (names.HYDROSEQ, names.DOWNSTREAM_HYDROSEQ, names.UPSTREAM_HYDROSEQ))
    return [Proposal('dropEmptyReaches', {}, f'drop the {int(empty.sum())} reaches with empty geometry'
                     + (', rewiring hydroseq links around them' if has_links else ''),
                     defect, preview=dict(reaches_dropped=int(empty.sum()), rewire_hydroseq=has_links))]


def _clearanceTable(rivers, reach_ids, opts) -> Dict[str, float]:
    id_col = opts['id_col']
    width = watershed_workflow.mesh.river_mesh.widthByOrderFunction(
        watershed_workflow.mesh.river_mesh.computeWidthByOrder(_live(rivers)))
    rows = rivers[rivers[id_col].astype(str).isin(reach_ids)]
    return { str(r[id_col]) : float(opts['clearance_factor'] * width(r)) for _, r in rows.iterrows() }


def _proposeOffsetDivide(defect, hucs, rivers, opts) -> List[Proposal]:
    """A reach crossing one divide several times: move the divide off it.

    The divide is the HUC boundary the reach crosses most; the HUC holding
    most of its length keeps it (the other way round is offered second).
    Other reaches running along the same divide are included so that one
    step fixes the whole run: those with at least 4 x clearance of their
    length within their clearance of it, and less than 1 x clearance
    deeper than that in the HUC giving up area.  (A stream simply flowing
    across the divide has about 2 x clearance near it and runs on into both
    HUCs, so it is left out.)
    """
    id_col, huc_id_col = opts['id_col'], opts['huc_id_col']
    live = _live(rivers)
    rid = _findReach(live, defect.reaches[0], id_col) if defect.reaches else None
    if rid is None:
        return []
    line = live.loc[live[id_col].astype(str) == rid].geometry.iloc[0]

    # the divide it crosses most
    near = hucs[hucs.geometry.intersects(line)]
    best = None
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            a, b = near.iloc[i], near.iloc[j]
            divide = a.geometry.boundary.intersection(b.geometry.boundary)
            if divide.is_empty or divide.length == 0:
                continue
            x = line.intersection(divide)
            n = 0 if x.is_empty else len(getattr(x, 'geoms', [x, ]))
            if best is None or n > best[0]:
                best = (n, a, b, divide)
    if best is None or best[0] < 2:
        return []
    _, a, b, divide = best
    if line.intersection(a.geometry).length < line.intersection(b.geometry).length:
        a, b = b, a

    proposals = []
    for keep, other in ((a, b), (b, a)):
        both = keep.geometry.union(other.geometry)
        cand = live[live.geometry.intersects(both)]
        clr = _clearanceTable(rivers, list(cand[id_col].astype(str)), opts)
        run = []
        for _, r in cand.iterrows():
            c = clr[str(r[id_col])]
            zone = divide.buffer(c)
            near = r.geometry.intersection(both).intersection(zone).length
            deep_in_other = r.geometry.intersection(other.geometry).difference(zone).length
            if str(r[id_col]) == rid or (near >= 4 * c and deep_in_other < c):
                run.append(str(r[id_col]))
        params = dict(keep=keep[huc_id_col], other=other[huc_id_col], reaches=run,
                      clearance={ r : clr[r] for r in run }, id_col=id_col, huc_id_col=huc_id_col)
        p = Proposal('offsetDivide', params,
                     f'move the {keep[huc_id_col]}|{other[huc_id_col]} divide into {other[huc_id_col]}, '
                     f'{min(params["clearance"].values()):.1f}-{max(params["clearance"].values()):.1f} '
                     f'({opts["clearance_factor"]} x bankfull width) off reaches {run}', defect)
        sel = live[live[id_col].astype(str).isin(run)]
        try:
            new_hucs, rep = watershed_workflow.hydro.watershed.offsetDivideFromReaches(
                hucs.copy(), sel, params['keep'], params['other'],
                lambda row: params['clearance'][str(row[id_col])], id_col=huc_id_col)
        except Exception as err:
            logging.info(f'offsetDivide {keep[huc_id_col]}<-{other[huc_id_col]} not offered: {err}')
            continue
        p.preview = dict(rep)
        old = other.geometry
        new = new_hucs.loc[new_hucs[huc_id_col] == other[huc_id_col]].geometry.iloc[0]
        p.geometry = gpd.GeoDataFrame(dict(role=['area moved', 'divide before', 'reaches']),
                                      geometry=[old.difference(new), divide, sel.geometry.union_all()],
                                      crs=hucs.crs)
        if keep is b:
            p.caveats.append(f'{keep[huc_id_col]} holds less of reach {rid} than {other[huc_id_col]}')
        proposals.append(p)
    proposals.sort(key=lambda p: (len(p.caveats), p.preview.get('area_moved', np.inf)))
    return proposals


def _proposeDropDisconnected(defect, hucs, rivers, opts) -> List[Proposal]:
    """A HUC with more than one outlet, where one outlet is a separate network.

    A small network not connected to the main one -- typically the clipped
    end of a neighboring basin's stream -- leaves the domain on its own and
    gives its HUC a second outlet.  Offer to drop it.  It is ranked first if
    its reachcodes are from a different HUC8 than the main network's.
    """
    id_col = opts['id_col']
    live = _live(rivers)
    trees = watershed_workflow.hydro.river.createRivers(live.copy(), method='geometry')
    trees.sort(key=lambda t: -len(t))
    tree_of = { str(n[id_col]) : k for k, t in enumerate(trees) for n in t }

    rc = opts['reachcode_col']
    if rc in live:
        main_codes = pd.Series([str(n[rc])[:8] for n in trees[0]])
        domain_huc8 = main_codes.mode().iloc[0]
    else:
        domain_huc8 = None

    proposals = []
    for outlet in defect.details.get('outlets', []):
        ks = { tree_of.get(_findReach(live, r, id_col)) for r in outlet['reaches'] }
        ks.discard(None)
        if len(ks) != 1 or 0 in ks:
            continue
        tree = trees[ks.pop()]
        ids = [str(n[id_col]) for n in tree]
        sel = live[live[id_col].astype(str).isin(ids)]
        foreign = None
        if domain_huc8 is not None:
            foreign = bool(all(not str(c).startswith(domain_huc8) for c in sel[rc]))
        p = Proposal('dropReaches', dict(reaches=ids, id_col=id_col),
                     f'drop the separate network of {len(ids)} reach(es) {ids} that leaves '
                     f'{defect.hucs[0]} into {outlet["into"]}', defect,
                     preview=dict(reaches_dropped=len(ids), length=float(sel.length.sum()),
                                  foreign=foreign, domain_huc8=domain_huc8),
                     geometry=gpd.GeoDataFrame(dict(role=['reaches dropped']),
                                               geometry=[sel.geometry.union_all()], crs=hucs.crs))
        if foreign is False:
            p.caveats.append(f'its reachcodes are from the domain HUC8 {domain_huc8}; it may be a real '
                             'stream disconnected by an error in the network')
        proposals.append(p)
    proposals.sort(key=lambda p: len(p.caveats))
    return proposals


#
# Cuts separating the networks of a HUC with more than one outlet
#
class _CutChecker:
    """Tests whether a line is an acceptable cut of a HUC.

    It must run inside the HUC from boundary to boundary, keep each reach's
    clearance, have its ends (new triple junctions) farther than
    junction_tol from any reach endpoint (else simplify() would snap them
    onto it) and junction_spacing from the HUC's existing junctions, and
    split the HUC into two pieces, one holding the arm network and the
    other the rest.
    """
    def __init__(self, poly, arm, rest, reaches, clearances, endpoints, junction_tol,
                 huc_junctions=(), junction_spacing=0., grid=1.e-3):
        self.poly = poly
        self.ring = poly.exterior
        self.cover = poly.buffer(10 * grid)
        self.arm, self.rest = arm, rest
        self.reaches, self.clearances = list(reaches), np.array(clearances)
        self.blocked = shapely.ops.unary_union([r.buffer(c) for r, c in zip(self.reaches, self.clearances)])
        blocked_ends = []
        if len(endpoints) > 0 and junction_tol > 0:
            blocked_ends.append(shapely.geometry.MultiPoint(endpoints).buffer(junction_tol))
        if len(huc_junctions) > 0 and junction_spacing > 0:
            blocked_ends.append(shapely.geometry.MultiPoint(huc_junctions).buffer(junction_spacing))
        self.junctions = shapely.ops.unary_union(blocked_ends) if blocked_ends else shapely.geometry.Polygon()
        self.grid = grid
        for g in (self.cover, self.blocked, self.junctions):
            shapely.prepare(g)

    def snapEnds(self, line):
        c = np.array(line.coords)
        for k in (0, -1):
            c[k] = self.ring.interpolate(self.ring.project(shapely.geometry.Point(c[k]))).coords[0]
        return shapely.geometry.LineString(c)

    def quick(self, lines):
        """Vectorized pre-check of an array of lines: inside, clear of reaches."""
        return shapely.covers(self.cover, lines) & ~shapely.intersects(self.blocked, lines)

    def check(self, line):
        """(arm piece, other piece) if line is an acceptable cut, else None."""
        if not (self.quick(np.array([line, ]))[0]):
            return None
        ends = shapely.points(np.array(line.coords)[[0, -1]])
        if shapely.intersects(self.junctions, ends).any():
            return None
        c = np.array(line.coords)
        d0, d1 = c[0] - c[1], c[-1] - c[-2]
        ext = shapely.geometry.LineString([c[0] + d0 / np.linalg.norm(d0)] + list(c) + [c[-1] + d1 / np.linalg.norm(d1)])
        pieces = [p for p in shapely.ops.split(self.poly, ext).geoms if p.area > 0]
        if len(pieces) != 2:
            return None
        k = int(np.argmax([self.arm.intersection(p).length for p in pieces]))
        arm_piece, other = pieces[k], pieces[1 - k]
        if self.arm.intersection(other).length > 1.e-6 or self.rest.intersection(arm_piece).length > 1.e-6:
            return None
        return arm_piece, other

    def clearanceRatio(self, line):
        return float(min(r.distance(line) / c for r, c in zip(self.reaches, self.clearances)))


def straightNeckCuts(checker : _CutChecker,
                     spacing : float = 20.,
                     max_length : Optional[float] = None,
                     neck_ratio : float = 3.,
                     max_cuts : int = 2,
                     max_length_factor : float = 2.) -> List[shapely.geometry.LineString]:
    """The shortest straight cuts across necks of the HUC, shortest first.

    Points every `spacing` along the boundary are paired if they are closer
    than max_length (default: the square root of the HUC's area) and the
    boundary between them, either way round, is at least neck_ratio times
    longer than the straight line -- i.e. the HUC is pinched there.  Pairs
    are tried shortest first.  Up to max_cuts distinct cuts are returned,
    none longer than max_length_factor times the shortest.
    """
    ring = checker.ring
    L = ring.length
    s = np.arange(0., L, spacing)
    pts = shapely.line_interpolate_point(ring, s)
    ok = ~shapely.intersects(checker.blocked, pts) & ~shapely.intersects(checker.junctions, pts)
    s, P = s[ok], shapely.get_coordinates(pts[ok])
    if len(P) < 2:
        return []
    if max_length is None:
        max_length = np.sqrt(checker.poly.area)

    pairs = cKDTree(P).query_pairs(max_length, output_type='ndarray')
    if len(pairs) == 0:
        return []
    chord = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
    arc = np.abs(s[pairs[:, 0]] - s[pairs[:, 1]])
    keep = np.minimum(arc, L - arc) >= neck_ratio * chord
    pairs, chord = pairs[keep], chord[keep]
    order = np.argsort(chord)
    pairs, chord = pairs[order], chord[order]

    cuts : List[shapely.geometry.LineString] = []
    ends : List[np.ndarray] = []
    batch = 5000
    for b in range(0, len(pairs), batch):
        if cuts and chord[b] > max_length_factor * cuts[0].length:
            break
        lines = shapely.linestrings(np.stack([P[pairs[b:b + batch, 0]], P[pairs[b:b + batch, 1]]], axis=1))
        for k in np.nonzero(checker.quick(lines))[0]:
            line = lines[k]
            if cuts and line.length > max_length_factor * cuts[0].length:
                return cuts
            e = np.array(line.coords)
            if any(min(np.linalg.norm(e - f, axis=1).max(), np.linalg.norm(e - f[::-1], axis=1).max())
                   < 10 * spacing for f in ends):
                continue
            if checker.check(line) is not None:
                cuts.append(line)
                ends.append(e)
                if len(cuts) == max_cuts:
                    return cuts
    return cuts


def equidistantCut(checker : _CutChecker,
                   spacing : float = 20.,
                   simplify_tols : Tuple[float, ...] = (40., 20., 10., 0.)) -> Optional[shapely.geometry.LineString]:
    """The line equidistant from the arm network and the rest of the HUC's reaches.

    Built from the Voronoi diagram of the reaches, sampled every `spacing`
    and labelled by network: the boundary of the arm's cells inside the
    HUC.  It has the largest possible clearance from both networks, but
    ignores topography.  It is simplified with the largest of
    simplify_tols that keeps it an acceptable cut.  None if the arm's
    region meets the HUC boundary in more than one stretch (one cut cannot
    separate it) or no simplification is acceptable.
    """
    def sample(g):
        if g.is_empty:
            return np.zeros((0, 2))
        return shapely.get_coordinates(shapely.segmentize(g, spacing))

    arm_pts = np.unique(sample(checker.arm), axis=0)
    rest_pts = np.unique(sample(checker.rest), axis=0)
    if len(arm_pts) == 0 or len(rest_pts) == 0:
        return None
    rest_set = set(map(tuple, rest_pts))
    arm_pts = np.array([p for p in arm_pts if tuple(p) not in rest_set])
    pts = np.concatenate([arm_pts, rest_pts])
    cells = shapely.voronoi_polygons(shapely.multipoints(pts),
                                     extend_to=checker.poly.envelope.buffer(checker.poly.length),
                                     ordered=True)
    cells = list(cells.geoms)
    region = shapely.ops.unary_union(cells[:len(arm_pts)]).intersection(checker.poly)
    if region.geom_type != 'Polygon':
        parts = list(getattr(region, 'geoms', []))
        parts = [p for p in parts if p.geom_type == 'Polygon']
        if not parts:
            return None
        region = max(parts, key=lambda p: checker.arm.intersection(p).length)

    inside = region.boundary.difference(checker.ring.buffer(10 * checker.grid))
    inside = shapely.ops.linemerge(inside) if inside.geom_type == 'MultiLineString' else inside
    if inside.geom_type != 'LineString':
        logging.info(f'equidistantCut: the arm region meets the boundary in more than one stretch')
        return None
    for tol in simplify_tols:
        line = checker.snapEnds(inside.simplify(tol) if tol > 0 else inside)
        if checker.check(line) is not None:
            return line
    return None


def _outletNetworks(poly, live, outlets, id_col):
    """For each outlet of a HUC, the reaches in the HUC that drain to it (River nodes)."""
    trees = watershed_workflow.hydro.river.createRivers(live.copy(), method='geometry')
    outlet_ids = [set(_findReach(live, r, id_col) for r in o['reaches']) - {None, } for o in outlets]
    groups : List[list] = [[] for _ in outlets]
    others = []
    for tree in trees:
        for node in tree:
            if node.linestring.intersection(poly).length == 0:
                continue
            n = node
            while n is not None:
                k = next((k for k, ids in enumerate(outlet_ids) if str(n[id_col]) in ids), None)
                if k is not None:
                    groups[k].append(node)
                    break
                n = n.parent
            else:
                others.append(node)
    return trees, groups, others


def _nextHUCId(hucs, huc, huc_id_col):
    taken = set(hucs[huc_id_col].astype(str))
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        if f'{huc}{letter}' not in taken:
            return f'{huc}{letter}'
    raise RuntimeError(f'no free ID for a piece of {huc}')


def _hucJunctions(hucs, poly):
    """Points on poly's boundary where its neighbor changes (HUC triple junctions)."""
    pts = []
    for g in hucs.geometry:
        if g.equals(poly) or not g.intersects(poly):
            continue
        inter = poly.boundary.intersection(g.boundary)
        lines = [l for l in getattr(inter, 'geoms', [inter, ]) if l.geom_type == 'LineString' and l.length > 0]
        if lines:
            merged = shapely.ops.linemerge(lines)
            pts.extend(map(tuple, shapely.get_coordinates(merged.boundary)))
    return pts


def _checkBuildsWatershed(hucs, changed, huc_id_col):
    """Raise if the changed HUCs and their neighbors do not build a Watershed (e.g. a pair of
    HUCs now sharing two separate stretches of boundary)."""
    sel = hucs[huc_id_col].isin([c for c in changed if c is not None])
    region = hucs.loc[sel].geometry.union_all().buffer(1.)
    local = hucs[hucs.geometry.intersects(region)]
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        watershed_workflow.hydro.watershed.Watershed(local.copy())
    finally:
        plt.show = show
        plt.close('all')


def _proposeSplitHUC(defect, hucs, rivers, opts) -> List[Proposal]:
    """A HUC with more than one outlet: cut each extra outlet's network off.

    For each outlet but the main one (largest drainage), find cuts that
    separate the reaches draining to it from the rest (straightNeckCuts,
    then equidistantCut), and offer for each cut: the piece as a new HUC,
    and the piece merged into the HUC it drains into.
    """
    id_col, huc_id_col = opts['id_col'], opts['huc_id_col']
    outlets = defect.details.get('outlets', [])
    if len(outlets) < 2 or not defect.hucs:
        return []
    huc = defect.hucs[0]
    poly = hucs.loc[hucs[huc_id_col] == huc].geometry.iloc[0]
    live = _live(rivers)
    trees, groups, others = _outletNetworks(poly, live, outlets, id_col)
    if sum(1 for g in groups if g) < 2:
        return []

    def size(g):
        das = [n[names.DRAINAGE_AREA] for n in g if names.DRAINAGE_AREA in n]
        return (max(das) if das else 0., sum(n.linestring.length for n in g))
    main = max(range(len(groups)), key=lambda k: size(groups[k]))

    width = watershed_workflow.mesh.river_mesh.widthByOrderFunction(
        watershed_workflow.mesh.river_mesh.computeWidthByOrder(live))
    clearance = lambda n: opts['clearance_factor'] * width(n)
    nodes = [n for t in trees for n in t]
    max_c = max(clearance(n) for n in nodes)
    near = [n for n in nodes if n.linestring.distance(poly) < max_c]
    endpoints = [n.linestring.coords[-1] for n in nodes] + [n.linestring.coords[0] for t in trees for n in t.leaf_nodes]
    endpoints = [p[0:2] for p in endpoints if shapely.geometry.Point(p).distance(poly) < opts['junction_tol']]

    proposals = []
    for k, group in enumerate(groups):
        if k == main or not group:
            continue
        arm = shapely.ops.unary_union([n.linestring for n in group]).intersection(poly)
        rest = shapely.ops.unary_union([n.linestring for j, g in enumerate(groups) if j != k for n in g]
                                       + [n.linestring for n in others]).intersection(poly)
        checker = _CutChecker(poly, arm, rest, [n.linestring for n in near], [clearance(n) for n in near],
                              endpoints, opts['junction_tol'], _hucJunctions(hucs, poly),
                              opts['junction_spacing'])
        cuts = [('straight', c) for c in straightNeckCuts(checker)]
        eq = equidistantCut(checker)
        if eq is not None:
            cuts.append(('equidistant', eq))
        if not cuts:
            logging.info(f'proposeFixes: no acceptable cut separates outlet {k} of HUC {huc}')
            continue

        into = outlets[k]['into']
        into_huc = None if into == 'domain exterior' else into
        arm_ids = [str(n[id_col]) for n in group]
        for method, line in cuts:
            arm_piece, other = checker.check(line)
            preview = dict(method=method, cut_length=line.length,
                           min_clearance_ratio=checker.clearanceRatio(line),
                           piece_area=arm_piece.area, remaining_area=other.area, arm_reaches=len(arm_ids))
            geometry = gpd.GeoDataFrame(dict(role=['cut', 'piece', 'arm reaches', 'other reaches']),
                                        geometry=[line, arm_piece, arm, rest], crs=hucs.crs)
            caveat = ('a straight line across the neck, not traced from topography' if method == 'straight'
                      else 'the line equidistant from the two networks, not traced from topography')
            base = dict(huc=huc, cut=[list(c) for c in line.coords],
                        piece_point=list(arm_piece.representative_point().coords[0]), id_col=huc_id_col)

            new_id = _nextHUCId(hucs, huc, huc_id_col)
            options = [(Proposal('splitHUC', dict(base, new_id=new_id, new_tohuc=into_huc),
                                 f'split HUC {huc} along a {line.length:.0f} {method} cut; the piece '
                                 f'drained by reaches {arm_ids[:3]}{"..." if len(arm_ids) > 3 else ""} '
                                 f'becomes HUC {new_id}, draining to {into}', defect,
                                 preview=dict(preview), geometry=geometry, caveats=[caveat, ]), 0)]
            if into_huc is not None:
                target = hucs.loc[hucs[huc_id_col] == into_huc].geometry.iloc[0]
                p = Proposal('splitHUC', dict(base, merge_into=into_huc),
                             f'cut the piece of HUC {huc} drained by reaches {arm_ids[:3]}'
                             f'{"..." if len(arm_ids) > 3 else ""} off along a {line.length:.0f} {method} '
                             f'cut and add it to {into_huc}', defect, preview=dict(preview), geometry=geometry,
                             caveats=[caveat, f'{into_huc} grows by {arm_piece.area / target.area:.0%} and '
                                      f'{huc} shrinks by {arm_piece.area / poly.area:.0%}'])
                options.append((p, 1))
            for p, rank in options:
                try:
                    _checkBuildsWatershed(p.apply(hucs, rivers)[0], [huc, into_huc, p.params.get('new_id')],
                                          huc_id_col)
                except Exception as err:
                    logging.info(f'proposeFixes: {p.description} not offered: {err}')
                    continue
                proposals.append((rank, len(proposals), p))
    return [p for _, _, p in sorted(proposals, key=lambda t: (t[0], t[1]))]


PROPOSERS : Dict[str, List[Callable[..., List[Proposal]]]] = {
    'empty-geometry' : [_proposeDropEmpty, ],
    'multiple-crossing' : [_proposeOffsetDivide, ],
    'multiple-outlets' : [_proposeDropDisconnected, _proposeSplitHUC],
}


def proposeFixes(report : DefectReport,
                 hucs : gpd.GeoDataFrame,
                 rivers : gpd.GeoDataFrame,
                 clearance_factor : float = 2.5,
                 reachcode_col : str = 'reachcode',
                 id_col : str = names.ID,
                 huc_id_col : str = names.ID,
                 recipe : Optional[Recipe] = None) -> List[List[Proposal]]:
    """Candidate fixes for each defect in a report, best first.

    Parameters
    ----------
    report : DefectReport
        From diagnostics.findDefects(hucs, rivers, ...).
    hucs, rivers : gpd.GeoDataFrame
        The same inputs the report was made from.  Not modified.
    clearance_factor : float, optional
        Divide clearance as a multiple of the reach's mean bankfull width
        for its stream order.
    reachcode_col : str, optional
        NHD reachcode column, used to tell a neighboring basin's stream.
    id_col, huc_id_col : str, optional
        ID columns of rivers and hucs.
    recipe : Recipe, optional
        Proposals it has already rejected are left out.

    Returns
    -------
    list of list of Proposal
        One list per report.defects entry; empty if nothing is proposed.
    """
    opts = dict(clearance_factor=clearance_factor, reachcode_col=reachcode_col, id_col=id_col,
                huc_id_col=huc_id_col,
                junction_tol=report.params.get('snap_triple_junctions_tol', 0.) or 0.,
                junction_spacing=report.params.get('reach_segment_target_length', 0.) or 0.)
    out = []
    for defect in report.defects:
        props = []
        for proposer in PROPOSERS.get(defect.kind, []):
            try:
                props.extend(proposer(defect, hucs, rivers, opts))
            except Exception as err:
                logging.warning(f'proposeFixes: {proposer.__name__} failed on [{defect.kind}] '
                                f'{defect.message}: {type(err).__name__}: {err}')
        if recipe is not None:
            props = [p for p in props if not recipe.isRejected(p)]
        out.append(props)
    return out
