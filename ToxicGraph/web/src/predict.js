import Plotly from 'plotly.js-dist-min';
import {
  APP, state, TASK_NAMES, DS_COLORS,
  isPredicting, setIsPredicting,
  chartRendered, setChartRendered,
  tableRendered, setTableRendered,
  syncHashToUrl,
} from './state.js';
import { init3dViewer } from './viewer.js';
import { historyAdd, bookmarkAdd } from './history.js';
import { fetchAndRenderActivity } from './activity.js';

// ── plotly chart ───────────────────────────────
const CHART_TOP_N = 15;
let _dsFilter = null; // null = all shown; Set = specific datasets

export function initDsFilterChips() {
  const row = document.getElementById('ds-filter-chips');
  if (!row || !APP.taskGroups) return;
  _dsFilter = null; // reset to "all" each time a new prediction runs
  const datasets = Object.keys(APP.taskGroups);
  row.style.display = 'flex';
  row.innerHTML = datasets.map(ds =>
    `<button class="ds-chip active" data-ds="${ds}"
       onclick="toggleDsFilter('${ds}',this)"
       style="--chip-col:${APP.dsColors?.[ds]||'#64748b'}">${ds}</button>`
  ).join('');
}

export function toggleDsFilter(ds, btn) {
  // Initialise from "all" on first toggle
  if (_dsFilter === null) {
    _dsFilter = new Set(Object.keys(APP.taskGroups || {}));
  }
  if (_dsFilter.has(ds)) { _dsFilter.delete(ds); btn.classList.remove('active'); }
  else                   { _dsFilter.add(ds);    btn.classList.add('active');    }
  renderChart(state.isTestSet || false, state.gt || null);
}

export function renderChart(isTestSet, gt) {
  const means = state.means || [];
  const stds  = state.stds  || [];
  if (!means.length) return;

  // Keep only top CHART_TOP_N tasks by probability, respecting dataset filter
  const topIdx = [...means.map((v, i) => ({v, i}))]
    .sort((a, b) => b.v - a.v)
    .filter(({i}) => _dsFilter === null || _dsFilter.has(APP.taskDatasets?.[i]))
    .slice(0, CHART_TOP_N).map(x => x.i)
    .sort((a, b) => a - b); // restore original order for grouping
  const topNames  = topIdx.map(i => TASK_NAMES[i]);
  const topMeans  = topIdx.map(i => means[i]);
  const topStds   = topIdx.map(i => stds[i] || 0);
  const colors  = topIdx.map(i => (DS_COLORS[i] || '#64748b') + 'cc');
  const borders = topIdx.map(i => DS_COLORS[i] || '#64748b');
  const modelLabel = (APP.models && APP.models[0]) ? APP.models[0].toUpperCase() : 'Model';

  const traces = [{
    type: 'bar', x: topNames, y: topMeans,
    error_y: {type:'data',array:topStds,visible:true,color:'#9b9590',thickness:1.2,width:3},
    marker: {color: colors, line: {color: borders, width: 1}},
    hovertemplate: '<b>%{x}</b><br>prob: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>',
    name: modelLabel,
  }];

  if (state.compareMode && state.meansB) {
    const meansB = topIdx.map(i => state.meansB[i]);
    const stdsB  = topIdx.map(i => (state.stdsB || [])[i] || 0);
    const labelB = (APP.models && APP.models[1]) ? APP.models[1].toUpperCase() : 'Model B';
    traces.push({
      type: 'bar', x: topNames, y: meansB,
      error_y: {type:'data',array:stdsB,visible:true,color:'#b45309',thickness:1.2,width:3},
      marker: {color: meansB.map(() => 'rgba(180,83,9,.35)'), line: {color: meansB.map(() => '#b45309'), width: 1}},
      hovertemplate: `<b>%{x}</b><br>${labelB}: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>`,
      name: labelB,
    });
  }

  const topIdxSet = new Set(topIdx);
  if (isTestSet && gt) {
    const showMissing = document.getElementById('show-missing-cb')?.checked || false;
    const gtX_pos=[], gtX_neg=[], gtX_mis=[];
    const gtY_pos=[], gtY_neg=[], gtY_mis=[];
    gt.forEach((v, i) => {
      if (!topIdxSet.has(i)) return;
      if (v===1)      { gtX_pos.push(TASK_NAMES[i]); gtY_pos.push(1.02); }
      else if (v===0) { gtX_neg.push(TASK_NAMES[i]); gtY_neg.push(1.02); }
      else            { gtX_mis.push(TASK_NAMES[i]); gtY_mis.push(1.02); }
    });
    traces.push({type:'scatter',mode:'markers',x:gtX_pos,y:gtY_pos,name:'GT positive',
      marker:{symbol:'diamond',size:8,color:'#dc2626'},hovertemplate:'<b>%{x}</b><br>GT: positive<extra></extra>'});
    traces.push({type:'scatter',mode:'markers',x:gtX_neg,y:gtY_neg,name:'GT negative',
      marker:{symbol:'circle',size:7,color:'#9b9590'},hovertemplate:'<b>%{x}</b><br>GT: negative<extra></extra>'});
    if (showMissing) {
      traces.push({type:'scatter',mode:'markers',x:gtX_mis,y:gtY_mis,name:'Missing',
        marker:{symbol:'x',size:6,color:'#b45309'},hovertemplate:'<b>%{x}</b><br>GT: not in dataset<extra></extra>'});
    }
  }

  const shapes = [], annotations = [];
  if (APP.taskGroups && APP.dsColors) {
    let offset = -0.5;
    Object.entries(APP.taskGroups).forEach(([ds, tasks]) => {
      const visibleCount = tasks.filter(t => topIdxSet.has(TASK_NAMES.indexOf(t))).length;
      if (!visibleCount) return;
      const end = offset + visibleCount;
      const col = APP.dsColors[ds] || '#64748b';
      shapes.push({type:'rect',x0:offset,x1:end,y0:0,y1:1,
        fillcolor:col+'08',line:{width:0},layer:'below'});
      annotations.push({x:(offset+end)/2,y:1.13,xref:'x',yref:'paper',text:ds,showarrow:false,
        font:{size:10,color:col}});
      offset = end;
    });
  }
  shapes.push({type:'line',x0:-0.5,x1:topNames.length-0.5,y0:.5,y1:.5,
    line:{color:'rgba(0,0,0,.1)',width:1,dash:'dot'}});

  const layout = {
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{t:14,b:140,l:34,r:10},
    height: 360,
    barmode: state.compareMode ? 'group' : 'relative',
    xaxis:{tickangle:-55,tickfont:{size:8,family:'JetBrains Mono',color:'#9b9590'},
           gridcolor:'rgba(0,0,0,.05)',zeroline:false},
    yaxis:{range:[0,1.12],gridcolor:'rgba(0,0,0,.06)',zeroline:false,
           tickfont:{size:10,family:'JetBrains Mono',color:'#9b9590'}},
    shapes, annotations,
    legend:{orientation:'h',y:-0.52,font:{size:10,family:'DM Sans'}},
    showlegend: isTestSet || state.compareMode,
    hovermode:'closest',
    font:{family:'DM Sans'},
  };
  Plotly.newPlot('task-chart', traces, layout, {responsive:true,displayModeBar:false});

  // task drill-down on bar click
  const chartEl = document.getElementById('task-chart');
  chartEl.on('plotly_click', async evtData => {
    const pt = evtData.points[0];
    if (!pt) return;
    const info = await loadTaskInfo();
    showTaskPopover(pt.x, info[pt.x] || null, evtData.event);
  });
}

// ── task info popover ──────────────────────────
let _taskInfo = null;
async function loadTaskInfo() {
  if (_taskInfo) return _taskInfo;
  try { _taskInfo = await fetch('/static/task_info.json').then(r => r.json()); }
  catch { _taskInfo = {}; }
  return _taskInfo;
}

export function showTaskPopover(taskName, info, evt) {
  let pop = document.getElementById('task-popover');
  if (!pop) { pop = document.createElement('div'); pop.id = 'task-popover'; document.body.appendChild(pop); }
  pop.className = 'task-popover';

  const badge = info
    ? `<span class="ds-pill ${info.dataset}" style="font-size:10px">${info.dataset}</span>`
    : '';
  pop.innerHTML = `
    <button class="tp-close" onclick="document.getElementById('task-popover').remove()">✕</button>
    <div class="tp-name">${taskName}</div>
    <div class="tp-label">${info ? info.label : taskName}</div>
    <div class="tp-desc">${info ? info.desc : 'No description available.'}</div>
    <div class="tp-footer">${badge}</div>`;

  const x = Math.min(evt.clientX + 12, window.innerWidth - 300);
  const y = Math.min(evt.clientY + 12, window.innerHeight - 200);
  pop.style.left = x + 'px';
  pop.style.top  = y + 'px';

  setTimeout(() => {
    document.addEventListener('click', function dismiss(e) {
      if (!pop.contains(e.target)) { pop.remove(); document.removeEventListener('click', dismiss); }
    });
  }, 0);
}

// ── comparison table ───────────────────────────
export function renderTable() {
  const means = state.means || [];
  if (!means.length) {
    document.getElementById('task-table-container').innerHTML =
      '<div style="color:var(--text-3);font-size:12px;padding:8px 0">Run a prediction to see results.</div>';
    return;
  }
  const meansB  = state.compareMode ? (state.meansB || null) : null;
  const gt      = state.isTestSet ? (state.gt || null) : null;
  const labelA  = (APP.models && APP.models[0]) ? APP.models[0].toUpperCase() : 'Model';
  const labelB  = (APP.models && APP.models[1]) ? APP.models[1].toUpperCase() : 'Model B';
  const indices = means.map((_, i) => i).sort((a, b) => means[b] - means[a]);

  let head = `<tr><th>#</th><th>Task</th><th>${labelA}</th>`;
  if (meansB) head += `<th>${labelB}</th><th>Δ</th>`;
  if (gt)     head += `<th>GT</th>`;
  head += '</tr>';

  const rows = indices.map((i, rank) => {
    const gnn   = means[i];
    const dmpnn = meansB ? meansB[i] : null;
    const delta = dmpnn != null ? dmpnn - gnn : null;
    const gtVal = gt ? gt[i] : null;
    const gnnCol   = gnn>=0.7?'var(--red)':gnn>=0.5?'var(--amber)':'var(--text-3)';
    const dmpnnCol = dmpnn!=null?(dmpnn>=0.7?'var(--red)':dmpnn>=0.5?'var(--amber)':'var(--text-3)'):'';
    const deltaCol = delta!=null?(delta>0.05?'var(--green-lt)':delta<-0.05?'var(--red)':'var(--text-3)'):'';
    let gtCell = '';
    if (gt) {
      if      (gtVal===1) gtCell = `<td><span style="color:var(--red);font-weight:700">◆ pos</span></td>`;
      else if (gtVal===0) gtCell = `<td><span style="color:var(--text-3)">● neg</span></td>`;
      else                gtCell = `<td><span style="color:var(--text-3);opacity:.5">—</span></td>`;
    }
    return `<tr>
      <td class="tc-rank">${rank+1}</td>
      <td class="tc-task">${TASK_NAMES[i]||i}</td>
      <td class="tc-prob" style="color:${gnnCol}">${gnn.toFixed(3)}</td>
      ${dmpnn!=null?`<td class="tc-prob" style="color:${dmpnnCol}">${dmpnn.toFixed(3)}</td>
        <td class="tc-delta" style="color:${deltaCol}">${delta>=0?'+':''}${delta.toFixed(3)}</td>`:''}
      ${gtCell}
    </tr>`;
  }).join('');

  document.getElementById('task-table-container').innerHTML =
    `<div class="tc-scroll"><table class="task-compare-table"><thead>${head}</thead><tbody>${rows}</tbody></table></div>`;
}

// ── summary bars ───────────────────────────────
export function updateSummaryBars(means, stds = []) {
  const top6 = [...means.map((v, i) => ({v, i}))]
    .sort((a, b) => b.v - a.v).slice(0, 4);
  const barsHtml = top6.map(({v, i}) => {
    const col = v>=0.7 ? 'var(--green)' : v>=0.4 ? 'var(--blue-lt)' : 'var(--text-3)';
    const std = stds[i];
    const badge = (std !== undefined && std > 0.12)
      ? `<span class="unc-badge" title="High uncertainty (MC std ${std.toFixed(3)})">?</span>`
      : '';
    return `<div class="task-row">
      <span class="task-name">${TASK_NAMES[i]||i}</span>
      <div class="task-track"><div class="task-fill" style="width:${v*100}%;background:${col}"></div></div>
      <span class="task-val" style="color:${col}">${v.toFixed(2)}${badge}</span>
    </div>`;
  }).join('');
  const container = document.getElementById('task-bars-container');
  container.innerHTML = barsHtml;
  container.querySelectorAll('.task-row').forEach((row, i) => {
    row.style.animationDelay = `${i * 18}ms`;
    row.classList.add('anim-fade-in');
  });

  // Star/bookmark button below the bars
  let starBtn = document.getElementById('summary-bookmark-btn');
  if (!starBtn) {
    starBtn = document.createElement('button');
    starBtn.id = 'summary-bookmark-btn';
    starBtn.className = 'bookmark-btn';
    starBtn.onclick = bookmarkCurrent;
    container.parentElement.appendChild(starBtn);
  }
  starBtn.innerHTML = '☆ Bookmark this prediction';
  starBtn.classList.remove('bookmarked');
}

export function bookmarkCurrent() {
  const smiles = document.getElementById('smiles-inp').value.trim();
  if (!smiles || !state.means) return;
  const topIdx = state.means.indexOf(Math.max(...state.means));
  bookmarkAdd({ smiles, topProb: state.means[topIdx], topTask: TASK_NAMES[topIdx] || '—', isTestSet: false });
  const btn = document.getElementById('summary-bookmark-btn');
  if (btn) { btn.innerHTML = '★ Bookmarked'; btn.classList.add('bookmarked'); }
}

// ── atom attribution ───────────────────────────
export async function explainTopTask() {
  const smiles = document.getElementById('smiles-inp').value.trim();
  const btn = document.getElementById('explain-btn');
  const taskIdx = parseInt(btn.dataset.taskIdx || '0', 10);
  const panel = document.getElementById('attr-panel');
  const container = document.getElementById('attr-svg-container');

  panel.style.display = '';
  container.innerHTML = '<div style="color:var(--text-3);font-size:11px;padding:16px 0">Computing… ~1s</div>';

  try {
    const r = await fetch(`/api/explain?smiles=${encodeURIComponent(smiles)}&task=${taskIdx}`);
    if (!r.ok) throw new Error((await r.json()).detail || 'Server error');
    const data = await r.json();
    document.getElementById('attr-task-name').textContent = data.task_name;
    container.innerHTML = data.svg;
    const svg = container.querySelector('svg');
    if (svg) { svg.style.width = '100%'; svg.style.height = 'auto'; }
  } catch (e) {
    container.innerHTML = `<div style="color:var(--red);font-size:11px;padding:16px 0">Failed: ${e.message}</div>`;
  }
}

export function closeAttr() {
  document.getElementById('attr-panel').style.display = 'none';
}

// ── csv export ─────────────────────────────────
export function exportCSV() {
  const means = state.means || [];
  if (!means.length) return;
  const meansB = state.compareMode ? state.meansB : null;
  const gt     = state.isTestSet   ? state.gt     : null;
  const smiles = document.getElementById('mol-smiles-display').textContent.trim();

  const headers = ['rank', 'task', (APP.models && APP.models[0]) ? APP.models[0].toUpperCase() : 'prob'];
  if (meansB) headers.push((APP.models && APP.models[1]) ? APP.models[1].toUpperCase() : 'prob_b', 'delta');
  if (gt)     headers.push('ground_truth');

  const indices = means.map((_, i) => i).sort((a, b) => means[b] - means[a]);
  const rows = indices.map((i, rank) => {
    const row = [rank+1, TASK_NAMES[i]||i, means[i].toFixed(4)];
    if (meansB) row.push(meansB[i].toFixed(4), (meansB[i]-means[i]).toFixed(4));
    if (gt)     row.push(gt[i]===1 ? 'positive' : gt[i]===0 ? 'negative' : 'missing');
    return row.join(',');
  });

  const csv  = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], {type:'text/csv'});
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `toxicgraph_${smiles.slice(0,20).replace(/[^a-zA-Z0-9]/g,'_')}.csv`;
  a.click();
}

// ── print report ───────────────────────────────
export function printReport() {
  const means = state.means || [];
  if (!means.length) return;
  const smiles = document.getElementById('smiles-inp').value.trim();
  const date   = new Date().toLocaleDateString();
  const model  = (APP.models && APP.activeModel) ? APP.activeModel.toUpperCase() : 'Ensemble';

  const rows = means
    .map((v, i) => ({v, i}))
    .sort((a, b) => b.v - a.v)
    .map(({v, i}) => {
      const std = state.stds?.[i];
      return `<tr>
        <td>${TASK_NAMES[i]||i}</td>
        <td>${(v*100).toFixed(1)}%</td>
        <td>${std !== undefined ? (std*100).toFixed(1)+'%' : '—'}</td>
      </tr>`;
    }).join('');

  const el = document.getElementById('print-report');
  el.innerHTML = `
    <div class="pr-header">
      <h1>ToxicGraph Toxicity Report</h1>
      <div class="pr-meta"><strong>SMILES:</strong> ${smiles}</div>
      <div class="pr-meta"><strong>Model:</strong> ${model} · <strong>Date:</strong> ${date}</div>
    </div>
    <table class="pr-table">
      <thead><tr><th>Task</th><th>Probability</th><th>MC Std</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="pr-footer">GNN + DMPNN ensemble · MC dropout · temperature-scaled calibration · ToxicGraph</div>`;
  window.print();
}

// ── fetch helper ───────────────────────────────
export async function fetchPrediction(smiles, model) {
  const r = await fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({smiles, n_mc: 20, model}),
  });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${r.status}`); }
  return r.json();
}

// ── molecular properties ───────────────────────
function renderProperties(p) {
  const el = document.getElementById('prop-strip');
  if (!p || !el) return;
  const badge = p.lipinski
    ? '<span class="prop-badge pass">Ro5 ✓</span>'
    : '<span class="prop-badge fail">Ro5 ✗</span>';
  el.innerHTML = [
    ['MW',   p.mw],
    ['LogP', p.logp],
    ['TPSA', p.tpsa],
    ['HBD',  p.hbd],
    ['HBA',  p.hba],
    ['RotB', p.rot_bonds],
    ['QED',  p.qed],
  ].map(([l, v]) =>
    `<div class="prop-item"><span class="prop-lbl">${l}</span><span class="prop-val">${v}</span></div>`
  ).join('') + `<div class="prop-item">${badge}</div>`;
  el.style.display = 'flex';
  el.classList.remove('anim-slide-up');
  void el.offsetWidth;
  el.classList.add('anim-slide-up');
}

// ── predict ────────────────────────────────────
export async function runPredict() {
  if (isPredicting) return;
  const smiles = document.getElementById('smiles-inp').value.trim();
  if (!smiles) {
    const bar = document.getElementById('nav-query-bar');
    bar.classList.add('error');
    setTimeout(() => bar.classList.remove('error'), 1800);
    return;
  }
  setIsPredicting(true);
  const queryBtn = document.querySelector('.query-btn');
  if (queryBtn) queryBtn.disabled = true;

  if (!document.getElementById('view-predict').classList.contains('active')) {
    window.showView('predict', document.querySelector('#view-toggle .vt-btn'));
  }

  const propStrip = document.getElementById('prop-strip');
  if (propStrip) propStrip.style.display = 'none';

  ['chip-max-auc','chip-mc-std','chip-top-task'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.add('skeleton');
    el.textContent = ' ';
  });

  document.getElementById('mol-placeholder').style.display = 'none';
  document.getElementById('predict-loader').style.display = 'flex';
  document.getElementById('predict-loader').innerHTML = '<div class="loader-ring"></div>predicting…';
  document.getElementById('mol-mode-badge').textContent = 'Predicting…';

  let data, dataB = null;
  const compareMode = APP.activeModel === null && APP.models && APP.models.length > 1;
  try {
    if (compareMode) {
      [data, dataB] = await Promise.all(APP.models.map(m => fetchPrediction(smiles, m)));
    } else {
      data = await fetchPrediction(smiles, APP.activeModel || undefined);
    }
  } catch(e) {
    document.getElementById('predict-loader').innerHTML =
      `<span style="color:#dc2626;font-size:11px">⚠ ${e.message}</span>`;
    document.getElementById('mol-mode-badge').textContent = 'Error';
    setIsPredicting(false);
    if (queryBtn) queryBtn.disabled = false;
    return;
  }

  document.getElementById('mol-smiles-display').textContent = smiles;
  const modelLabel = compareMode
    ? APP.models.map(m => m.toUpperCase()).join(' vs ')
    : data.model_used.toUpperCase();
  document.getElementById('mol-name-display').textContent =
    `New prediction · ${modelLabel} · click and drag to rotate`;
  document.getElementById('mol-mode-badge').textContent = compareMode ? 'Comparing models' : 'New prediction';
  document.getElementById('mol-mode-badge').className = 'mch-badge new';
  document.getElementById('gt-legend').style.display = 'none';
  document.getElementById('new-mol-note').style.display = 'flex';
  document.getElementById('missing-toggle-wrap').style.display = 'none';

  const topProbA = data.max_auc.toFixed(3);
  const topProbDisplay = compareMode && dataB ? `${topProbA} / ${dataB.max_auc.toFixed(3)}` : topProbA;
  document.getElementById('chip-max-auc').textContent = topProbDisplay;
  document.getElementById('chip-max-auc').style.fontSize = compareMode && dataB ? '16px' : '';
  const lbl = document.getElementById('chip-max-auc-lbl');
  if (lbl) lbl.textContent = compareMode && dataB
    ? `Top Prob (${APP.models.map(m => m.toUpperCase()).join(' / ')})`
    : 'Top Prob';
  document.getElementById('chip-mc-std').textContent  = data.mc_std_mean.toFixed(3);
  document.getElementById('chip-top-task').textContent = data.top_task;
  ['chip-max-auc','chip-mc-std','chip-top-task'].forEach(id =>
    document.getElementById(id).classList.remove('skeleton'));
  document.querySelectorAll('.metric-chip').forEach((el, i) => {
    el.classList.remove('anim-slide-up');
    void el.offsetWidth;
    el.style.animationDelay = `${i * 60}ms`;
    el.classList.add('anim-slide-up');
  });

  state.means        = data.means;
  state.stds         = data.stds;
  state.meansB       = dataB ? dataB.means : null;
  state.stdsB        = dataB ? dataB.stds  : null;
  state.gt           = null;
  state.isTestSet    = false;
  state.compareMode  = compareMode;
  setChartRendered(false);
  setTableRendered(false);

  if (!document.getElementById('tp-chart').classList.contains('hidden')) {
    renderChart(false, null);
    setChartRendered(true);
  }
  if (!document.getElementById('tp-table').classList.contains('hidden')) {
    renderTable();
    setTableRendered(true);
  }
  updateSummaryBars(data.means, data.stds);

  // show explain button for top-scoring task
  const topIdx = data.means.indexOf(Math.max(...data.means));
  const explainBtn = document.getElementById('explain-btn');
  if (explainBtn) {
    explainBtn.dataset.taskIdx = topIdx;
    explainBtn.style.display = '';
    document.getElementById('attr-panel').style.display = 'none';
  }

  if (data.sdf) {
    if (smiles !== state.renderedSmiles) {
      state.renderedSmiles = smiles;
      init3dViewer(document.getElementById('predict-3d'), data.sdf);
    }
    document.getElementById('predict-loader').style.display = 'none';
  } else {
    document.getElementById('predict-loader').innerHTML =
      '<span style="color:#9b9590;font-size:11px">3D unavailable</span>';
  }

  document.getElementById('tp-export-btn').style.display = '';
  const reportBtn = document.getElementById('tp-report-btn');
  if (reportBtn) reportBtn.style.display = '';
  initDsFilterChips();
  syncHashToUrl(smiles);
  historyAdd({smiles, topProb: data.max_auc, topTask: data.top_task, isTestSet: false});

  // non-blocking parallel fetches — properties and activity predictions
  fetch(`/api/properties/${encodeURIComponent(smiles)}`)
    .then(r => r.ok ? r.json() : null)
    .then(renderProperties)
    .catch(() => {});

  fetchAndRenderActivity(smiles);

  setIsPredicting(false);
  if (queryBtn) queryBtn.disabled = false;
}
