// Activity view — RF fingerprint predictions for BBBP and HIV.
// Renders both the mini-card in the predict view and the full dedicated activity page.

let _activityAvailable = false;

export function setActivityAvailable(flag) {
  _activityAvailable = flag;
}

// ── mini-card (predict view) ───────────────────────────────────────────────────

export async function fetchAndRenderActivity(smiles) {
  if (!_activityAvailable) return;

  const card = document.getElementById('activity-card');
  if (!card) return;

  card.style.display = '';
  card.querySelectorAll('.act-prob').forEach(el => {
    el.textContent = '—';
    el.classList.add('skeleton');
  });
  card.querySelectorAll('.act-fill').forEach(el => { el.style.width = '0%'; });
  card.querySelectorAll('.act-ep-status').forEach(el => { el.textContent = ''; });

  try {
    const r = await fetch(`/api/activity?smiles=${encodeURIComponent(smiles)}`);
    if (!r.ok) { _renderMiniError(card); return; }
    const data = await r.json();
    _renderMiniCard(card, data);
  } catch {
    _renderMiniError(card);
  }
}

function _renderMiniCard(card, data) {
  data.tasks.forEach(task => {
    const entry    = data.predictions[task.key];
    const prob     = typeof entry === 'object' ? entry.prob : entry;
    const active   = prob >= 0.5;
    const probEl   = document.getElementById(`act-${task.key}-prob`);
    const fillEl   = document.getElementById(`act-${task.key}-fill`);
    const statusEl = document.getElementById(`act-${task.key}-status`);
    if (!probEl) return;

    probEl.classList.remove('skeleton');
    probEl.textContent = prob.toFixed(3);
    probEl.style.color = active ? task.color : 'var(--text-3)';
    fillEl.style.width      = `${prob * 100}%`;
    fillEl.style.background = active ? task.color : 'var(--surface2)';
    statusEl.textContent = active ? 'Active' : 'Inactive';
    statusEl.style.color = active ? task.color : 'var(--text-3)';
  });

  card.classList.remove('anim-slide-up');
  void card.offsetWidth;
  card.classList.add('anim-slide-up');
}

function _renderMiniError(card) {
  card.querySelectorAll('.act-prob').forEach(el => {
    el.classList.remove('skeleton');
    el.textContent = '—';
  });
}

// ── full activity page ─────────────────────────────────────────────────────────

export async function runActivityPredict() {
  if (!_activityAvailable) return;
  const inp = document.getElementById('act-smiles-inp');
  const smiles = inp?.value.trim();
  if (!smiles) {
    const bar = document.getElementById('act-query-bar');
    if (bar) { bar.classList.add('error'); setTimeout(() => bar.classList.remove('error'), 1800); }
    return;
  }

  const resultsEl = document.getElementById('act-results');
  resultsEl.innerHTML = '<div class="act-page-loading">Running RF prediction…</div>';

  try {
    const r = await fetch(`/api/activity?smiles=${encodeURIComponent(smiles)}`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      resultsEl.innerHTML = `<div class="act-page-error">${err.detail || 'Prediction failed'}</div>`;
      return;
    }
    const data = await r.json();
    _renderActivityPage(smiles, data, resultsEl);
  } catch (e) {
    resultsEl.innerHTML = `<div class="act-page-error">Network error: ${e.message}</div>`;
  }
}

export function prefillActivitySmiles(smiles) {
  const inp = document.getElementById('act-smiles-inp');
  if (inp) inp.value = smiles;
}

function _renderActivityPage(smiles, data, container) {
  const cards = data.tasks.map(task => {
    const entry  = data.predictions[task.key];
    const prob   = typeof entry === 'object' ? entry.prob : entry;
    const std    = typeof entry === 'object' ? entry.std  : 0;
    const active = prob >= 0.5;
    const pct    = Math.round(prob * 100);
    const color  = task.color;
    const gaugeStyle = `background:conic-gradient(${color} 0% ${pct}%,var(--surface2) ${pct}% 100%)`;

    return `
      <div class="act-page-card">
        <div class="act-page-card-top">
          <span class="act-page-task-name">${task.label}</span>
          <span class="act-page-badge ${active ? 'active' : 'inactive'}">${active ? 'Active' : 'Inactive'}</span>
        </div>
        <div class="act-page-gauge-wrap">
          <div class="act-page-gauge" style="${gaugeStyle}">
            <div class="act-page-gauge-inner">
              <div class="act-page-prob" style="color:${active ? color : 'var(--text-3)'}">${prob.toFixed(3)}</div>
              <div class="act-page-prob-lbl">probability</div>
            </div>
          </div>
        </div>
        <div class="act-page-task-desc">${task.desc}</div>
        <div class="act-page-uncertainty">
          <span class="act-page-unc-label">Tree uncertainty</span>
          <div class="act-page-unc-bar-wrap">
            <div class="act-page-unc-bar" style="width:${Math.min(std * 400, 100)}%;background:${color}55"></div>
          </div>
          <span class="act-page-unc-val" style="color:${color}">\xB1${std.toFixed(3)}</span>
        </div>
      </div>`;
  }).join('');

  const chartId = 'act-chart-' + Date.now();

  container.innerHTML = `
    <div class="act-page-smiles-row">
      <span class="act-page-smiles-label">SMILES</span>
      <span class="act-page-smiles-val">${smiles}</span>
    </div>
    <div class="act-page-cards">${cards}</div>
    <div class="act-page-chart-wrap">
      <div class="act-page-chart-title">Probability with tree-vote uncertainty (\xB11 std)</div>
      <div id="${chartId}" style="width:100%;height:220px"></div>
    </div>
    <div class="act-page-meta">Morgan(512) + MACCS(167) \xB7 300-tree RandomForest per task \xB7 uncertainty = std of individual tree votes</div>
  `;

  if (window.Plotly) {
    const tasks  = data.tasks;
    const probs  = tasks.map(t => { const e = data.predictions[t.key]; return typeof e === 'object' ? e.prob : e; });
    const stds   = tasks.map(t => { const e = data.predictions[t.key]; return typeof e === 'object' ? e.std  : 0; });
    const colors = tasks.map(t => t.color);
    const labels = tasks.map(t => t.label);

    window.Plotly.newPlot(chartId, [{
      type: 'bar',
      x: labels,
      y: probs,
      error_y: { type: 'data', array: stds, visible: true, color: '#94a3b8', thickness: 2, width: 10 },
      marker: { color: colors, opacity: 0.85 },
      hovertemplate: '%{x}<br>prob: %{y:.3f} \xB1 %{error_y.array:.3f}<extra></extra>',
    }], {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      margin: { t: 10, r: 20, b: 60, l: 50 },
      yaxis: {
        range: [0, 1],
        gridcolor: '#334155',
        tickfont: { size: 11 },
        title: { text: 'Probability', font: { size: 11 } },
      },
      xaxis: { tickfont: { size: 13, family: 'Inter' } },
      font:  { family: 'Inter, sans-serif', color: '#94a3b8' },
      shapes: [{
        type: 'line', x0: -0.5, x1: labels.length - 0.5, y0: 0.5, y1: 0.5,
        line: { color: '#475569', width: 1, dash: 'dot' },
      }],
      annotations: [{
        x: labels.length - 0.5, y: 0.5, xanchor: 'right', yanchor: 'bottom',
        text: 'decision boundary', showarrow: false,
        font: { size: 9, color: '#475569' },
      }],
    }, { responsive: true, displayModeBar: false });
  }
}
