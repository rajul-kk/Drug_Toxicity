// Shared mutable state — replaces window._xxx globals and top-level var declarations.
// APP and state are objects so their properties are directly mutable by any importer.
// Primitive re-exports (let) require setter functions so live bindings propagate.

export const APP = {};

export const state = {
  means:          null,
  stds:           null,
  meansB:         null,
  stdsB:          null,
  gt:             null,
  isTestSet:      false,
  compareMode:    false,
  renderedSmiles: null,
};

export let TASK_NAMES    = [];
export let DS_COLORS     = [];
export let isPredicting  = false;
export let chartRendered = false;
export let tableRendered = false;

export function setTaskNames(v)     { TASK_NAMES     = v; }
export function setDSColors(v)      { DS_COLORS      = v; }
export function setIsPredicting(v)  { isPredicting   = v; }
export function setChartRendered(v) { chartRendered  = v; }
export function setTableRendered(v) { tableRendered  = v; }

// URL hash helpers live here to avoid circular imports between app.js and predict.js
export function syncHashToUrl(smiles) {
  const hash = smiles ? '#predict?s=' + encodeURIComponent(smiles) : '#';
  history.replaceState(null, '', hash);
}
