const https = require('https');

https.get('https://unpkg.com/deck.gl@8.9.34/dist.min.js', (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
        console.log("Downloaded " + data.length + " bytes.");
        // A hacky way to see if _GlobeView or GlobeView exists in the bundle text
        console.log("Contains _GlobeView:", data.includes('_GlobeView'));
        console.log("Contains GlobeView:", data.includes('GlobeView'));
    });
});
