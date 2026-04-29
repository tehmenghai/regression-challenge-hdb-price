/**
 * HDB Knowledge Base — Top Navigation Bar
 * Injected into every doc page. Works locally (file://) and on CloudFront (https://).
 * To update the "Best score" badge, change BEST_SCORE below.
 */
(function () {
  'use strict';

  var BEST_SCORE = '$21,335';

  // Works for file:// (local) and https:// (CloudFront)
  var homeUrl = 'index.html';

  /* ── Styles ── */
  var style = document.createElement('style');
  style.textContent = [
    '#hdb-nav {',
    '  position:fixed;top:0;left:0;right:0;z-index:9999;',
    '  background:#0f172a;color:#e2e8f0;',
    '  height:48px;display:flex;align-items:center;',
    '  padding:0 20px;gap:14px;',
    '  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;',
    '  font-size:0.88rem;box-shadow:0 2px 8px rgba(0,0,0,.35);',
    '}',
    '#hdb-nav a{color:#e2e8f0;text-decoration:none;}',
    '#hdb-nav a:hover{color:#fff;}',
    '#hdb-nav .hn-home{',
    '  background:#0369a1;padding:5px 13px;border-radius:6px;',
    '  font-weight:600;white-space:nowrap;',
    '}',
    '#hdb-nav .hn-home:hover{background:#0284c7;}',
    '#hdb-nav .hn-sep{color:#334155;font-size:1.1em;}',
    '#hdb-nav .hn-title{flex:1;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}',
    '#hdb-nav .hn-score{',
    '  background:#d97706;color:#fff;padding:3px 11px;',
    '  border-radius:12px;font-weight:700;font-size:0.8rem;white-space:nowrap;',
    '}',
  ].join('');
  document.head.appendChild(style);

  /* ── Push body down so fixed bar doesn't overlap content ── */
  document.addEventListener('DOMContentLoaded', function () {
    document.body.style.paddingTop = '60px';

    var h1 = document.querySelector('h1');
    var title = h1 ? h1.textContent.trim() : document.title;
    if (title.length > 65) title = title.slice(0, 62) + '…';

    var nav = document.createElement('div');
    nav.id = 'hdb-nav';
    nav.innerHTML =
      '<a href="' + homeUrl + '" class="hn-home">&#127968; Knowledge Base</a>' +
      '<span class="hn-sep">&#8250;</span>' +
      '<span class="hn-title">' + title + '</span>' +
      '<span class="hn-score">&#127942; Best: ' + BEST_SCORE + '</span>';

    document.body.insertBefore(nav, document.body.firstChild);
  });
})();
