import {
  APP, state, TASK_NAMES,
  chartRendered, setChartRendered,
  tableRendered, setTableRendered,
} from './state.js';
import { renderChart, renderTable, updateSummaryBars } from './predict.js';
import { init3dViewer } from './viewer.js';
import { historyAdd } from './history.js';

export const browseState = {filter: 'all', sort: 'conf', page: 1, mode: 'browse', smarts: ''};

let _searchDebounce = null;

function skeletonRows() {
  const widths = [200, 240, 180, 220, 195];
  return widths.map(w => `
    <tr class="sk-row">
      <td class="thumb-cell"><div class="skeleton sk-cell" style="width:44px;height:44px;border-radius:8px"></div></td>
      <td><div class="skeleton sk-cell" style="width:${w}px"></div></td>
      <td><div class="skeleton sk-cell" style="width:52px;border-radius:999px"></div></td>
      <td><div class="skeleton sk-cell" style="width:100px"></div></td>
      <td><div class="skeleton sk-cell" style="width:36px"></div></td>
      <td><div class="skeleton sk-cell" style="width:48px;border-radius:8px"></div></td>
    </tr>`).join('');
}

export function onSearchInput(val) {
  clearTimeout(_searchDebounce);
  if (!val.trim()) {
    browseState.mode = 'browse';
    browseState.page = 1;
    fetchBrowse();
    return;
  }
  _searchDebounce = setTimeout(() => {
    browseState.mode   = 'search';
    browseState.smarts = val.trim();
    browseState.page   = 1;
    fetchSearch();
  }, 400);
}

export async function fetchBrowse() {
  const tbody = document.getElementById('mol-tbody');
  tbody.innerHTML = skeletonRows();
  try {
    const {filter, sort, page} = browseState;
    const url = `/api/testset?model=${APP.activeModel||''}&page=${page}&per_page=20&filter=${filter}&sort=${sort}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderBrowseRows(data.rows);
    updatePagination(data.page, data.pages, data.total, false);
    document.getElementById('browse-sub').textContent =
      `${data.total} molecules · click any row to load in viewer`;
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--red);font-size:12px">Failed to load: ${e.message}</td></tr>`;
    document.getElementById('browse-sub').textContent = 'Error loading test set';
  }
}

async function fetchSearch() {
  const tbody = document.getElementById('mol-tbody');
  tbody.innerHTML = skeletonRows();
  try {
    const {smarts, page} = browseState;
    const url = `/api/search?smarts=${encodeURIComponent(smarts)}&model=${APP.activeModel||''}&page=${page}&per_page=20`;
    const r = await fetch(url);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    renderBrowseRows(data.rows);
    updatePagination(data.page, data.pages, data.total, true);
    document.getElementById('browse-sub').textContent = data.total > 0
      ? `${data.total} match${data.total===1?'':'es'} for "${smarts}"`
      : `No matches for "${smarts}"`;
  } catch(e) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--red);font-size:12px">Search error: ${e.message}</td></tr>`;
    document.getElementById('browse-sub').textContent = 'Search error';
  }
}

function renderBrowseRows(rows) {
  const tbody = document.getElementById('mol-tbody');
  tbody.innerHTML = '';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-3);font-size:12px">No results.</td></tr>';
    return;
  }
  rows.forEach((r, rowIdx) => {
    const conf = r.max_conf;
    const col  = conf>=0.8 ? 'var(--green)' : conf>=0.5 ? 'var(--amber)' : 'var(--blue-lt)';
    const tr   = document.createElement('tr');
    tr.innerHTML = `
      <td class="thumb-cell">
        <img class="thumb-img" src="/api/thumbnail/${encodeURIComponent(r.smiles)}?size=80"
          onerror="this.style.display='none'" alt="" loading="lazy">
      </td>
      <td class="smiles-cell">${r.smiles}</td>
      <td><span class="ds-pill ${r.dataset}">${r.dataset}</span></td>
      <td>
        <div class="conf-bar-wrap">
          <div class="conf-track"><div class="conf-fill" style="width:${conf*100}%;background:${col}"></div></div>
          <span class="conf-val" style="color:${col}">${conf.toFixed(2)}</span>
        </div>
      </td>
      <td><span class="badge ${r.score>=0.75?'g':r.score>=0.55?'b':'a'}">${r.score.toFixed(2)}</span></td>
      <td><button class="open-btn" onclick="openFromBrowse(${r.idx})">Open →</button></td>`;
    tr.style.animationDelay = `${rowIdx * 25}ms`;
    tr.classList.add('anim-fade-in');
    tbody.appendChild(tr);
  });
}

function updatePagination(page, pages, total, isSearch) {
  document.getElementById('page-info').textContent =
    `Showing ${(page-1)*20+1}–${Math.min(page*20,total)} of ${total}`;
  const btns = document.getElementById('page-btns');
  btns.innerHTML = '';
  const navigate = p => { browseState.page = p; isSearch ? fetchSearch() : fetchBrowse(); };
  const makeBtn  = (label, p) => {
    const b = document.createElement('div');
    b.className = 'page-btn' + (p===page ? ' active' : '');
    b.textContent = label;
    if (p && p !== page) b.onclick = () => navigate(p);
    btns.appendChild(b);
  };
  if (pages <= 7) {
    for (let i=1; i<=pages; i++) makeBtn(i, i);
  } else {
    makeBtn(1, 1);
    if (page > 3) makeBtn('…', null);
    for (let i=Math.max(2,page-1); i<=Math.min(pages-1,page+1); i++) makeBtn(i, i);
    if (page < pages-2) makeBtn('…', null);
    makeBtn(pages, pages);
  }
}

export function setFilter(name, el) {
  document.querySelectorAll('.fchip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  browseState.filter = name;
  browseState.page   = 1;
  browseState.mode   = 'browse';
  const inp = document.getElementById('smarts-inp');
  if (inp) inp.value = '';
  fetchBrowse();
}

export function sortTable(key) {
  browseState.sort = key;
  browseState.page = 1;
  browseState.mode === 'search' ? fetchSearch() : fetchBrowse();
}

export async function openFromBrowse(idx) {
  window.showView('predict', document.querySelector('#view-toggle .vt-btn'));

  document.getElementById('mol-placeholder').style.display = 'none';
  document.getElementById('predict-loader').style.display = 'flex';
  document.getElementById('predict-loader').innerHTML = '<div class="loader-ring"></div>loading…';

  const propStrip = document.getElementById('prop-strip');
  if (propStrip) propStrip.style.display = 'none';

  let data;
  try {
    const r = await fetch(`/api/testset/${idx}?model=${APP.activeModel||''}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch(e) {
    document.getElementById('predict-loader').innerHTML =
      `<span style="color:var(--red);font-size:11px">⚠ ${e.message}</span>`;
    return;
  }

  const needsRender = data.smiles !== state.renderedSmiles;

  const topProb = Math.max(...data.probs);
  const topIdx  = data.probs.indexOf(topProb);
  document.getElementById('chip-max-auc').textContent = topProb.toFixed(3);
  document.getElementById('chip-max-auc').style.fontSize = '';
  const lbl = document.getElementById('chip-max-auc-lbl');
  if (lbl) lbl.textContent = 'Top Prob';
  document.getElementById('chip-mc-std').textContent  = '0.000';
  document.getElementById('chip-top-task').textContent = TASK_NAMES[topIdx] || '—';

  state.compareMode   = false;
  state.meansB        = null;
  state.stdsB         = null;
  state.renderedSmiles = data.smiles;
  state.means         = data.probs;
  state.stds          = data.probs.map(() => 0);
  state.gt            = data.labels;
  state.isTestSet     = true;

  document.getElementById('mol-smiles-display').textContent = data.smiles;
  document.getElementById('mol-name-display').textContent =
    `Test set · ${data.dataset} · ${data.model_used.toUpperCase()} · drag to rotate`;
  document.getElementById('mol-mode-badge').textContent = 'Test set';
  document.getElementById('mol-mode-badge').className   = 'mch-badge test';
  document.getElementById('gt-legend').style.display       = 'flex';
  document.getElementById('new-mol-note').style.display    = 'none';
  document.getElementById('missing-toggle-wrap').style.display = 'flex';

  setChartRendered(false);
  setTableRendered(false);
  if (!document.getElementById('tp-chart').classList.contains('hidden')) {
    renderChart(true, data.labels);
    setChartRendered(true);
  }
  if (!document.getElementById('tp-table').classList.contains('hidden')) {
    renderTable();
    setTableRendered(true);
  }

  updateSummaryBars(data.probs);
  document.getElementById('tp-export-btn').style.display = '';
  historyAdd({smiles: data.smiles, topProb, topTask: TASK_NAMES[topIdx]||'—',
              isTestSet: true, dataset: data.dataset});

  if (data.sdf) {
    if (needsRender) init3dViewer(document.getElementById('predict-3d'), data.sdf);
    document.getElementById('predict-loader').style.display = 'none';
  } else {
    document.getElementById('predict-loader').innerHTML =
      '<span style="color:#9b9590;font-size:11px">3D unavailable</span>';
  }
}
