// Entry point — imports all modules and exposes functions used by
// inline HTML event handlers (onclick="fn()") and cross-module window bridges.

import './particles.js';
import { init3dViewer }    from './viewer.js';
import { runPredict, exportCSV, showTaskPopover, explainTopTask, closeAttr } from './predict.js';
import { fetchBrowse, setFilter, sortTable, openFromBrowse, onSearchInput } from './browse.js';
import { historyToggle, historyOpen } from './history.js';
import { runBatch }        from './multi.js';
import {
  showView, switchTaskTab, copySmiles, toggleMissingLabels, setModel, toggleTheme, boot,
} from './app.js';

// Expose to HTML inline handlers and cross-module window bridges
Object.assign(window, {
  // navigation & UI
  showView, switchTaskTab, copySmiles, toggleMissingLabels, setModel, toggleTheme,
  // predict view
  runPredict, exportCSV, showTaskPopover, explainTopTask, closeAttr,
  // browse view
  setFilter, sortTable, openFromBrowse, onSearchInput,
  // history
  historyToggle, historyOpen,
  // batch
  runBatch,
  // 3D viewer (used by boot() for hero molecule)
  init3dViewer,
});

boot();
