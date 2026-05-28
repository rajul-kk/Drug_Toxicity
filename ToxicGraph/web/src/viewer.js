import * as $3Dmol from '3dmol';

export function init3dViewer(el, sdf, spinSpeed) {
  spinSpeed = spinSpeed === undefined ? 0.4 : spinSpeed;
  el.innerHTML = '';
  const v = $3Dmol.createViewer(el, {backgroundColor: '#f0f4ff'});
  v.addModel(sdf, 'sdf');
  v.setStyle({}, {stick:{radius:.12,colorscheme:'Jmol'}, sphere:{scale:.22,colorscheme:'Jmol'}});
  v.zoomTo(); v.spin('y', spinSpeed); v.render();
  return v;
}
