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

import watershed_workflow.hydro.river
import watershed_workflow.hydro.watershed
import watershed_workflow.mesh.river_mesh
import watershed_workflow.sources.standard_names as names
from watershed_workflow.diagnostics import Defect, DefectReport

__all__ = ['Proposal', 'Recipe', 'proposeFixes', 'applyRecipe', 'ACTIONS', 'PROPOSERS',
           'dropEmptyReaches', 'dropReaches', 'offsetDivide']

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


ACTIONS : Dict[str, Callable[..., Frames]] = {
    'dropEmptyReaches' : dropEmptyReaches,
    'dropReaches' : dropReaches,
    'offsetDivide' : offsetDivide,
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


PROPOSERS : Dict[str, List[Callable[..., List[Proposal]]]] = {
    'empty-geometry' : [_proposeDropEmpty, ],
    'multiple-crossing' : [_proposeOffsetDivide, ],
    'multiple-outlets' : [_proposeDropDisconnected, ],
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
                huc_id_col=huc_id_col)
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
