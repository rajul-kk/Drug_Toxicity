// ── batch multi-molecule comparison ────────────
async function runBatch(){
  const raw = (document.getElementById('batch-inp')?.value || '').trim();
  if(!raw) return;

  const smilesList = raw.split('\n').map(s=>s.trim()).filter(Boolean);
  if(!smilesList.length) return;

  const btn = document.querySelector('.batch-run-btn');
  if(btn) btn.disabled = true;

  const resultsEl = document.getElementById('batch-results');
  resultsEl.innerHTML = `<div style="color:var(--text-3);font-size:12px;padding:8px 0">Running ${smilesList.length} prediction${smilesList.length>1?'s':''}…</div>`;

  // chunk into groups of 5 to respect rate limit
  const CHUNK = 5;
  const results = [];
  for(let i = 0; i < smilesList.length; i += CHUNK){
    const chunk = smilesList.slice(i, i + CHUNK);
    const chunkResults = await Promise.all(
      chunk.map(s =>
        fetchPrediction(s, APP.activeModel || undefined)
          .then(d => ({smiles: s, means: d.means, ok: true}))
          .catch(() => ({smiles: s, means: null, ok: false}))
      )
    );
    results.push(...chunkResults);
  }

  if(btn) btn.disabled = false;
  renderHeatmap(results);
}

function renderHeatmap(results){
  const resultsEl = document.getElementById('batch-results');
  if(!results.length){ resultsEl.innerHTML = ''; return; }

  const validResults = results.filter(r => r.ok && r.means);
  if(!validResults.length){
    resultsEl.innerHTML = '<div style="color:var(--red);font-size:12px;padding:8px 0">All predictions failed.</div>';
    return;
  }

  const tasks = TASK_NAMES;

  // header row with task names
  const headerCells = tasks.map(t =>
    `<th class="bh-col-header" title="${t}">${t}</th>`
  ).join('');

  // data rows
  const dataRows = results.map(r => {
    if(!r.ok || !r.means){
      const label = r.smiles.length > 20 ? r.smiles.slice(0,20)+'…' : r.smiles;
      return `<tr>
        <td class="bh-smiles-label" title="${r.smiles}">${label}</td>
        <td colspan="${tasks.length}" style="font-size:11px;color:var(--red);padding:0 8px">prediction failed</td>
      </tr>`;
    }
    const label = r.smiles.length > 20 ? r.smiles.slice(0,20)+'…' : r.smiles;
    const cells = r.means.map(prob => {
      const hue = Math.round((1 - prob) * 120); // 120=green, 0=red
      const bg = `hsl(${hue},65%,${55 + prob*0}%)`;
      return `<td class="bh-cell" style="background:${bg}" title="${r.smiles} · ${tasks[r.means.indexOf(prob)]||''} · ${prob.toFixed(3)}"></td>`;
    }).join('');
    return `<tr>
      <td class="bh-smiles-label" title="${r.smiles}">${label}</td>
      ${cells}
    </tr>`;
  }).join('');

  resultsEl.innerHTML = `
    <div class="batch-heatmap-wrap">
      <table class="batch-heatmap">
        <thead><tr><th class="bh-smiles-label"></th>${headerCells}</tr></thead>
        <tbody>${dataRows}</tbody>
      </table>
    </div>
    <div style="font-size:11px;color:var(--text-3);margin-top:8px">
      Hover cells for probability · red = high · green = low
    </div>`;
}
