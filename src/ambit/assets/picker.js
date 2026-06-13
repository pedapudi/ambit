
/* ===== Console colour-scheme picker — 16 themes ported from variants/T ===== */
const THEMES=[
  ["monokai","monokai",["#1e1f1c","#272822","#f8f8f2","#a6e22e","#f92672","#66d9ef"]],
  ["solarized-dark","solarized dark",["#04222B","#0A2D38","#93A1A1","#8BB80E","#E0483C","#2AA198"]],
  ["solarized-light","solarized light",["#FDF6E3","#FBF1D6","#586E75","#6B9B0B","#DC322F","#268BD2"]],
  ["google-light","google light",["#FFFFFF","#F4F4F4","#474A4E","#34A853","#EA4335","#1B9CB8"]],
  ["google-dark","google dark",["#202124","#2C2D30","#FFFFFF","#34A853","#EA4335","#24C1E0"]],
  ["lunaria-light","lunaria light",["#EBE4E1","#E2DCD9","#363434","#497D46","#783C1F","#3778A9"]],
  ["lunaria-eclipse","lunaria eclipse",["#323F46","#3B484F","#DFE2ED","#BEDBC1","#BA9088","#C8429F"]],
  ["belafonte-day","belafonte day",["#D5CCBA","#CCC3B2","#34292D","#6e6a4e","#BE100E","#426A79"]],
  ["belafonte-night","belafonte night",["#20111B","#271821","#D5CCBA","#a6a07a","#d6403e","#6F8E97"]],
  ["paper","paper",["#F2EEDE","#E6E2D3","#1A1A1A","#216609","#CC3E28","#1E6FCC"]],
  ["zenburn","zenburn",["#3A3A3A","#424241","#DCDCCC","#8FB28F","#CC9393","#8CD0D3"]],
  ["selenized-black","selenized black",["#181818","#202020","#DEDEDE","#83C746","#FF5E56","#56D8C9"]],
  ["relaxed","relaxed",["#353A44","#3D424B","#F7F7F7","#A0AC77","#BC5653","#7EAAC7"]],
  ["espresso","espresso",["#323232","#3A3A3A","#FFFFFF","#A5C261","#D25252","#6C99BB"]],
  ["dracula","dracula",["#282A36","#343746","#F8F8F2","#50FA7B","#FF5555","#BD93F9"]],
  ["ubuntu","ubuntu",["#300A24","#3D1530","#EEEEEC","#8AE234","#CC0000","#34E2E2"]],
];
const DEFAULT_THEME = "monokai";
const THEME_IDS = THEMES.map(t => t[0]);
function strip(swatches, sm) {
  return `<span class="dt-swatch-strip${sm?' dt-swatch-strip-sm':''}" aria-hidden="true">` +
    swatches.map(c => `<span class="dt-swatch" style="background:${c}"></span>`).join('') + `</span>`;
}
function mountThemePicker(hostId) {
  const host = document.getElementById(hostId);
  let value = DEFAULT_THEME, open = false, activeIdx = THEME_IDS.indexOf(value);
  const byId = new Map(THEMES.map(t => [t[0], t]));
  host.innerHTML =
    `<div class="dt-cd" role="group" aria-label="Colour theme">
      <button class="dt-cd-trigger" type="button" aria-haspopup="listbox" aria-expanded="false" aria-label="Colour theme" title="Colour theme">
        <span class="dt-cd-trig-strip"></span><span class="dt-cd-name dt-cd-trig-name"></span>
        <span class="dt-cd-caret" aria-hidden="true">▾</span>
      </button>
      <div class="dt-cd-list" role="listbox" aria-label="Colour theme">
        ${THEMES.map(([id,label,sw]) =>
          `<div class="dt-cd-option" role="option" data-theme="${id}" aria-selected="${id===value}" tabindex="-1" title="colour: ${label}">
             ${strip(sw)}<span class="dt-cd-name">${label}</span>
           </div>`).join('')}
      </div>
    </div>`;
  const cd = host.querySelector('.dt-cd');
  const trigger = host.querySelector('.dt-cd-trigger');
  const trigStrip = host.querySelector('.dt-cd-trig-strip');
  const trigName = host.querySelector('.dt-cd-trig-name');
  const list = host.querySelector('.dt-cd-list');
  const options = Array.from(host.querySelectorAll('.dt-cd-option'));
  function applyTheme(id) {
    value = THEME_IDS.includes(id) ? id : DEFAULT_THEME;
    document.documentElement.setAttribute('data-theme', value);
    const def = byId.get(value);
    trigStrip.className = 'dt-cd-trig-strip dt-swatch-strip dt-swatch-strip-sm';
    trigStrip.setAttribute('aria-hidden', 'true');
    trigStrip.innerHTML = def[2].map(c => `<span class="dt-swatch" style="background:${c}"></span>`).join('');
    trigName.textContent = def[1];
    options.forEach(o => o.setAttribute('aria-selected', String(o.getAttribute('data-theme') === value)));
  }
  function setActive(i) {
    activeIdx = (i + options.length) % options.length;
    options.forEach((o,k) => o.classList.toggle('dt-cd-active', k === activeIdx));
  }
  function setOpen(next) {
    open = next; cd.classList.toggle('dt-cd-open', open);
    trigger.setAttribute('aria-expanded', String(open));
    if (open) setActive(Math.max(0, THEME_IDS.indexOf(value)));
  }
  function choose(id) { applyTheme(id); setOpen(false); }
  options.forEach(o => o.addEventListener('click', () => choose(o.getAttribute('data-theme'))));
  trigger.addEventListener('click', () => setOpen(!open));
  trigger.addEventListener('keydown', ev => {
    if (['ArrowDown','Enter',' ','Spacebar'].includes(ev.key)) { ev.preventDefault(); setOpen(true); list.focus?.(); }
  });
  list.setAttribute('tabindex','-1');
  list.addEventListener('keydown', ev => {
    if (ev.key === 'Escape') { ev.preventDefault(); setOpen(false); trigger.focus(); }
    else if (ev.key === 'ArrowDown') { ev.preventDefault(); setActive(activeIdx+1); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); setActive(activeIdx-1); }
    else if (['Enter',' ','Spacebar'].includes(ev.key)) {
      ev.preventDefault();
      const id = options[activeIdx]?.getAttribute('data-theme'); if (id) choose(id);
    }
  });
  document.addEventListener('click', ev => {
    if (!open) return; let n = ev.target; while (n) { if (n === cd) return; n = n.parentNode; } setOpen(false);
  });
  applyTheme(DEFAULT_THEME);
}
mountThemePicker('theme-picker');
