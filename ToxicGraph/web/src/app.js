import {
  APP, state, TASK_NAMES,
  setTaskNames, setDSColors,
  chartRendered, setChartRendered,
  tableRendered, setTableRendered,
  syncHashToUrl,
} from './state.js';
import { fetchBrowse } from './browse.js';
import { historyRender } from './history.js';
import { renderChart, renderTable, runPredict } from './predict.js';
import { setActivityAvailable } from './activity.js';

// ── view switching ─────────────────────────────
export function showView(name, btn) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('#view-toggle .vt-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-'+name).classList.add('active');
  if (btn) btn.classList.add('active');
  document.body.classList.toggle('landing-mode', name === 'home');
  document.body.classList.toggle('app-mode', name !== 'home');
  document.getElementById('nav-query-bar').style.display = name === 'home' ? 'none' : '';
  const mt = document.getElementById('model-toggle');
  if (mt) mt.style.display = (name === 'predict' && APP.models && APP.models.length > 1) ? '' : 'none';
  if (name === 'home') syncHashToUrl('');
  if (name === 'browse' && APP.taskNames) fetchBrowse();
}

// ── task tab switching ─────────────────────────
export function switchTaskTab(name, btn) {
  document.querySelectorAll('.tp-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tp-body').forEach(b => b.classList.add('hidden'));
  btn.classList.add('active');
  document.getElementById('tp-'+name).classList.remove('hidden');
  if (name === 'chart' && !chartRendered) {
    renderChart(state.isTestSet || false, state.gt || null);
    setChartRendered(true);
  }
  if (name === 'table' && !tableRendered) {
    renderTable();
    setTableRendered(true);
  }
}

export function toggleMissingLabels() {
  setChartRendered(false);
  renderChart(state.isTestSet || false, state.gt || null);
  setChartRendered(true);
}

export function copySmiles(el) {
  const text = el.textContent.trim();
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const orig = el.textContent;
    el.textContent = 'Copied!';
    el.style.color = 'var(--green)';
    setTimeout(() => { el.textContent = orig; el.style.color = ''; }, 1200);
  });
}

// ── model toggle ───────────────────────────────
export function setModel(m, btn) {
  APP.activeModel = m;
  document.querySelectorAll('#model-toggle .vt-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const smiles = document.getElementById('smiles-inp').value.trim();
  if (smiles) runPredict();
}

// ── dark mode toggle ───────────────────────────
export function toggleTheme() {
  const dark = document.body.classList.toggle('dark');
  document.getElementById('theme-btn').textContent = dark ? '☽' : '☀';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}

// ── hash helpers ───────────────────────────────
function loadFromHash() {
  const m = location.hash.match(/[#?&]s=([^&]+)/);
  if (!m) return;
  const smiles = decodeURIComponent(m[1]);
  document.getElementById('smiles-inp').value = smiles;
  showView('predict', document.querySelector('#view-toggle .vt-btn'));
  runPredict();
}

// ── boot ───────────────────────────────────────
export async function boot() {
  if (localStorage.getItem('theme') === 'dark') {
    document.body.classList.add('dark');
    document.getElementById('theme-btn').textContent = '☽';
  }
  try {
    const r = await fetch('/api/info');
    if (!r.ok) throw new Error(`/api/info returned HTTP ${r.status}`);
    const info = await r.json();

    setTaskNames(info.task_names);
    setActivityAvailable(info.activity_available || false);
    APP.taskNames      = info.task_names;
    APP.taskGroups     = info.task_groups;
    APP.dsColors       = info.dataset_colors;
    APP.models         = info.available_models;
    APP.activeModel    = info.available_models.length > 1 ? null : info.default_model;
    APP.activityTasks  = info.activity_tasks || [];

    // populate model toggle
    const mt = document.getElementById('model-toggle');
    if (mt && APP.models.length > 1) {
      mt.innerHTML = APP.models.map(m =>
        `<button class="vt-btn" onclick="setModel('${m}',this)">${m.toUpperCase()}</button>`
      ).join('') + `<button class="vt-btn active" onclick="setModel(null,this)">Both</button>`;
    }

    const dsColArr = [];
    for (const [ds, tasks] of Object.entries(info.task_groups)) {
      const col = info.dataset_colors[ds] || '#64748b';
      tasks.forEach(() => dsColArr.push(col));
    }
    setDSColors(dsColArr);

    const chips = document.getElementById('filter-chips');
    chips.innerHTML = `<div class="fchip active all" onclick="setFilter('all',this)">All</div>` +
      Object.keys(info.task_groups).map(ds =>
        `<div class="fchip ${ds}" onclick="setFilter('${ds}',this)">${ds}</div>`
      ).join('');

    document.getElementById('hs-tasks').textContent    = info.task_names.length;
    document.getElementById('hs-datasets').textContent = Object.keys(info.task_groups).length;
    document.getElementById('hs-models').textContent   = '×' + info.available_models.length;
    document.getElementById('hero-eyebrow').textContent =
      info.available_models.map(m => m.toUpperCase()).join(' + ') +
      ' · ' + Object.keys(info.task_groups).length + ' datasets · ' +
      info.task_names.length + ' tasks';

    fetch('/api/testset/0?model=' + APP.activeModel)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        document.getElementById('hero-loader').style.display = 'none';
        const el = document.getElementById('hero3d');
        if (d && d.sdf && el) { window.init3dViewer(el, d.sdf, 0.3); }
      })
      .catch(() => { document.getElementById('hero-loader').style.display = 'none'; });

    fetchBrowse();
    historyRender();
    loadFromHash();
  } catch(e) {
    console.error('boot failed:', e);
    document.getElementById('browse-sub').textContent = 'Error: ' + e.message;
    document.getElementById('mol-tbody').innerHTML =
      `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--red);font-size:12px">App failed to initialise: ${e.message}</td></tr>`;
  }
}
