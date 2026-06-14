export async function fetchSimilar(smiles, n = 8) {
  const r = await fetch(`/api/similar?smiles=${encodeURIComponent(smiles)}&n=${n}`);
  if (!r.ok) throw new Error('Similar fetch failed');
  return r.json();
}

export function renderSimilarResults(results, containerEl) {
  if (!results.length) {
    containerEl.innerHTML = '<div style="color:var(--text-3);font-size:12px">No similar molecules found.</div>';
    return;
  }
  containerEl.innerHTML = results.map(({smiles, tanimoto}) => {
    const safe = smiles.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    return `<div class="similar-item" onclick="window.runPredictSmiles('${safe}')">
      <img class="similar-thumb"
           src="/api/thumbnail/${encodeURIComponent(smiles)}?size=48" alt=""
           onerror="this.style.display='none'">
      <div class="similar-info">
        <div class="similar-smiles">${smiles.length>26 ? smiles.slice(0,26)+'…' : smiles}</div>
        <div class="similar-sim">Tanimoto <strong>${tanimoto.toFixed(3)}</strong></div>
      </div>
    </div>`;
  }).join('');
}
