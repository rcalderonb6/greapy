// Load GoatCounter without auto-tracking (we handle it manually for SPA navigation)
var gc = document.createElement('script');
gc.setAttribute('data-goatcounter', 'https://greapy.goatcounter.com/count');
gc.setAttribute('data-goatcounter-settings', JSON.stringify({ no_onload: true }));
gc.src = '//gc.zgo.at/count.js';

gc.onload = function () {
    // Fire on initial load and each instant-navigation page change
    document$.subscribe(function () {
        window.goatcounter.count({
            path: location.pathname + location.search + location.hash,
        });
    });
};

document.head.appendChild(gc);
