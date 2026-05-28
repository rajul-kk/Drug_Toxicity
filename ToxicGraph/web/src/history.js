// history.js — localStorage prediction history
// showView and runPredict are accessed via window (set by main.js) to avoid circular imports

const HISTORY_KEY = 'tg_history';
const HISTORY_MAX = 20;

export function historyAdd(entry) {
  const list = historyLoad();
  const filtered = list.filter(e => e.smiles !== entry.smiles);
  filtered.unshift({...entry, ts: Date.now()});
  localStorage.setItem(HISTORY_KEY, JSON.stringify(filtered.slice(0, HISTORY_MAX)));
  historyRender();
}

export function historyLoad() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}

export function historyRender() {
  const list = historyLoad();
  const el = document.getElementById('history-list');
  if (!el) return;

  const btn = document.getElementById('hist-toggle-btn');
  if (btn) btn.textContent = list.length ? `History (${list.length})` : 'History';

  if (!list.length) {
    el.innerHTML = '<div class="hist-empty">No predictions yet.</div>';
    return;
  }

  el.innerHTML = list.map((e, i) => {
    const probCol = e.topProb >= 0.7 ? 'var(--red)' : e.topProb >= 0.4 ? 'var(--amber)' : 'var(--text-3)';
    const badge = e.isTestSet
      ? `<span class="ds-pill ${e.dataset||''}" style="font-size:9px">${e.dataset||'test'}</span>`
      : `<span class="mch-badge new" style="font-size:9px;padding:1px 6px">new</span>`;
    return `<div class="hist-item" onclick="historyOpen(${i})" title="${e.smiles}">
      <img class="hist-thumb" src="/api/thumbnail/${encodeURIComponent(e.smiles)}?size=60"
           onerror="this.style.display='none'" alt="" loading="lazy">
      <div class="hist-info">
        <div class="hist-smiles">${e.smiles.length>22 ? e.smiles.slice(0,22)+'…' : e.smiles}</div>
        <div class="hist-meta">
          ${badge}
          <span style="font-family:var(--mono);font-size:10px;color:${probCol}">${e.topProb.toFixed(3)}</span>
        </div>
        <div class="hist-task">${e.topTask||'—'}</div>
      </div>
    </div>`;
  }).join('');
}

export function historyOpen(i) {
  const list = historyLoad();
  const entry = list[i];
  if (!entry) return;
  historyToggle();
  document.getElementById('smiles-inp').value = entry.smiles;
  window.showView('predict', document.querySelector('#view-toggle .vt-btn'));
  window.runPredict();
}

export function historyToggle() {
  const panel = document.getElementById('history-panel');
  if (!panel) return;
  panel.classList.toggle('hidden');
}
