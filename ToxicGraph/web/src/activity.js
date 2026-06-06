// Activity profiling — RF fingerprint predictions for BBBP and HIV.
// Rendered as a card below the toxicity task panel in the predict view.

let _activityAvailable = false;

export function setActivityAvailable(flag) {
  _activityAvailable = flag;
}

export async function fetchAndRenderActivity(smiles) {
  if (!_activityAvailable) return;

  const card = document.getElementById('activity-card');
  if (!card) return;

  // Show skeleton state while loading
  card.style.display = '';
  card.querySelectorAll('.act-prob').forEach(el => {
    el.textContent = '—';
    el.classList.add('skeleton');
  });
  card.querySelectorAll('.act-fill').forEach(el => { el.style.width = '0%'; });
  card.querySelectorAll('.act-ep-status').forEach(el => { el.textContent = ''; });

  try {
    const r = await fetch(`/api/activity?smiles=${encodeURIComponent(smiles)}`);
    if (!r.ok) {
      _renderError(card);
      return;
    }
    const data = await r.json();
    _renderActivity(card, data);
  } catch {
    _renderError(card);
  }
}

function _renderActivity(card, data) {
  data.tasks.forEach(task => {
    const prob     = data.predictions[task.key];
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

function _renderError(card) {
  card.querySelectorAll('.act-prob').forEach(el => {
    el.classList.remove('skeleton');
    el.textContent = '—';
  });
}
