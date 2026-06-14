// Entry point — imports all modules and exposes functions used by
// inline HTML event handlers (onclick="fn()") and cross-module window bridges.

import './particles.js';
import { init3dViewer }    from './viewer.js';
import { runPredict, exportCSV, printReport, showTaskPopover, explainTopTask, closeAttr, toggleDsFilter, bookmarkCurrent } from './predict.js';
import { fetchBrowse, setFilter, sortTable, openFromBrowse, onSearchInput } from './browse.js';
import { historyToggle, historyOpen, historyOpenSmiles, bookmarkRemove } from './history.js';
import { runBatch, initCsvDrop } from './multi.js';
import { fetchSimilar, renderSimilarResults } from './similar.js';
import {
  showView, switchTaskTab, copySmiles, toggleMissingLabels, setModel, toggleTheme, boot,
} from './app.js';

// Expose to HTML inline handlers and cross-module window bridges
Object.assign(window, {
  // navigation & UI
  showView, switchTaskTab, copySmiles, toggleMissingLabels, setModel, toggleTheme,
  // predict view
  runPredict, exportCSV, printReport, showTaskPopover, explainTopTask, closeAttr, toggleDsFilter, bookmarkCurrent,
  // browse view
  setFilter, sortTable, openFromBrowse, onSearchInput,
  // history + bookmarks
  historyToggle, historyOpen, historyOpenSmiles, bookmarkRemove,
  // batch
  runBatch,
  // 3D viewer (used by boot() for hero molecule)
  init3dViewer,
});

boot();
initCsvDrop();

window.runPredictSmiles = smiles => {
  document.getElementById('smiles-inp').value = smiles;
  window.showView('predict', document.querySelector('#view-toggle .vt-btn'));
  window.runPredict();
};
