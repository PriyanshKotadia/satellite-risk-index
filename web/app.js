const {Deck, _GlobeView, ScatterplotLayer, SolidPolygonLayer, BitmapLayer} = deck;

// Earth radius in km
const EARTH_RADIUS = 6371;

// State
let allSatellites = [];
let appTime = 0;
let animationFrame = null;
let selectedConstellation = null;
let selectedSatId = null;
let trackedSatellite = null; // Cache the tracked satellite object
let hoveredSatId = null;
let isTracking = false;

// View State (Controlled)
let viewState = {
    longitude: 0,
    latitude: 0,
    zoom: 0,
    maxZoom: 10,
    minZoom: -10
};

// UI Elements
const searchInput = document.getElementById('search-input');
const clearSearchBtn = document.getElementById('clear-search-btn');
const detailsPanel = document.getElementById('details-panel');
const closePanelBtn = document.getElementById('close-panel-btn');
const highlightConstelBtn = document.getElementById('highlight-constel-btn');
const riskBar = document.getElementById('risk-bar');
const riskValue = document.getElementById('risk-value');

// Procedural Starfield
function initStarfield() {
    const canvas = document.getElementById('starfield');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw random stars
    const numStars = 1500;
    for (let i = 0; i < numStars; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        const radius = Math.random() * 1.5;
        // Distant stars are dimmer
        const opacity = Math.random() * 0.8 + 0.2; 
        
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
        ctx.fill();
    }
}
initStarfield();
window.addEventListener('resize', initStarfield);

// Initialize Deck.GL
const deckgl = new Deck({
    canvas: 'deck-canvas',
    width: '100%',
    height: '100%',
    viewState: viewState,
    onViewStateChange: ({viewState: newViewState, interactionState}) => {
        // Only stop tracking if the user is actively holding/dragging the globe.
        // This prevents residual pan momentum (isPanning) from instantly killing tracking.
        if (interactionState.isDragging) {
            isTracking = false;
        }
        viewState = newViewState;
        // MUST pass viewState back so DeckGL doesn't lose manual zoom changes
        deckgl.setProps({viewState});
    },
    controller: true,
    pickingRadius: 15, // Greatly increases the hit-area for moving targets
    views: new _GlobeView(),
    layers: [],
    getCursor: ({isHovering}) => isHovering ? 'pointer' : 'grab',
    onClick: (info) => {
        if (info && info.object) {
            selectSatellite(info.object);
        } else {
            // Clicked empty space
            clearSelection();
        }
    }
});

// Load Data
fetch('assets/predictions.json')
    .then(res => res.json())
    .then(data => {
        // Sort by risk so we retain the most important satellites if we downsample
        data.sort((a, b) => b.predicted_R - a.predicted_R);
        
        // Mobile Performance Cap: 17k is too heavy for many mobile GPUs.
        const isMobile = window.innerWidth < 768 || /Mobi|Android/i.test(navigator.userAgent);
        const MAX_MOBILE_SATS = 3000;
        
        if (isMobile && data.length > MAX_MOBILE_SATS) {
            allSatellites = data.slice(0, MAX_MOBILE_SATS);
            console.log(`Mobile device detected. Capped satellites to ${MAX_MOBILE_SATS} highest-risk objects for performance.`);
        } else {
            allSatellites = data;
        }
        
        console.log(`Loaded ${allSatellites.length} satellites.`);
        startAnimation();
    });

// Color mapping based on R
function getRiskColor(r) {
    if (r < 50) {
        const t = r / 50;
        return [34 + t * (245 - 34), 211 + t * (158 - 211), 238 + t * (11 - 238), 200];
    } else {
        const t = (r - 50) / 50;
        return [245 + t * (244 - 245), 158 + t * (63 - 158), 11 + t * (94 - 11), 200];
    }
}

// Helper to convert deg to rad
const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

function calculatePosition(sat, timeDelta) {
    const u = (sat.mean_anomaly + sat.mean_motion * timeDelta * 360) * D2R;
    const i = sat.inclination * D2R;
    const raan = sat.raan * D2R;
    
    const lat = Math.asin(Math.sin(i) * Math.sin(u));
    let lon = raan + Math.atan2(Math.cos(i) * Math.sin(u), Math.cos(u));
    
    return [lon * R2D, lat * R2D, sat.altitude_km * 1000];
}

function lerpAngle(a, b, t) {
    let diff = b - a;
    while (diff < -180) diff += 360;
    while (diff > 180) diff -= 360;
    return a + diff * t;
}

function renderLayers() {
    const searchTerm = searchInput.value.trim().toLowerCase();
    const isFiltered = selectedConstellation !== null || selectedSatId !== null || searchTerm !== "";
    
    // Background Earth Polygon (base sphere)
    const backgroundLayer = new SolidPolygonLayer({
        id: 'earth-base',
        data: [[[-180, 90], [0, 90], [180, 90], [180, -90], [0, -90], [-180, -90]]],
        getPolygon: d => d,
        stroked: false,
        filled: true,
        getFillColor: [10, 20, 40, 255] 
    });

    // Translucent textured earth mapping
    const earthTextureLayer = new BitmapLayer({
        id: 'earth-texture',
        bounds: [-180, -90, 180, 90],
        image: 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg',
        opacity: 0.4,
        transparentColor: [0, 0, 0, 0]
    });

    const satLayer = new ScatterplotLayer({
        id: 'satellites',
        data: allSatellites,
        pickable: true,
        radiusMinPixels: 1.5,
        radiusMaxPixels: 8, // Allow slightly larger visual radius on hover
        getPosition: d => calculatePosition(d, appTime),
        getFillColor: d => {
            const baseColor = getRiskColor(d.predicted_R);
            if (!isFiltered) {
                return hoveredSatId === d.norad_id ? [255, 255, 255, 255] : baseColor; // White flash on hover
            }
            
            let match = false;
            if (selectedConstellation && d.satellite_constellation === selectedConstellation) match = true;
            if (selectedSatId && d.norad_id === selectedSatId) match = true;
            
            const nameStr = (d.name || "").toString().toLowerCase();
            const constelStr = (d.satellite_constellation || "").toString().toLowerCase();
            const countryStr = (d.country || "").toString().toLowerCase();
            if (searchTerm && (nameStr.includes(searchTerm) || constelStr.includes(searchTerm) || countryStr.includes(searchTerm))) match = true;
            
            if (hoveredSatId === d.norad_id) return [255, 255, 255, 255];
            
            return match ? [baseColor[0], baseColor[1], baseColor[2], 255] : [baseColor[0], baseColor[1], baseColor[2], 20];
        },
        getRadius: d => {
            if (!isFiltered) {
                return hoveredSatId === d.norad_id ? 25000 : 10000;
            }
            
            let match = false;
            if (selectedConstellation && d.satellite_constellation === selectedConstellation) match = true;
            if (selectedSatId && d.norad_id === selectedSatId) match = true;
            
            const nameStr = (d.name || "").toString().toLowerCase();
            const constelStr = (d.satellite_constellation || "").toString().toLowerCase();
            const countryStr = (d.country || "").toString().toLowerCase();
            if (searchTerm && (nameStr.includes(searchTerm) || constelStr.includes(searchTerm) || countryStr.includes(searchTerm))) match = true;
            
            if (hoveredSatId === d.norad_id) return match ? 70000 : 25000;
            
            return match ? 50000 : 5000;
        },
        onHover: (info) => {
            const newHoveredId = (info && info.object) ? info.object.norad_id : null;
            if (newHoveredId !== hoveredSatId) {
                hoveredSatId = newHoveredId;
                renderLayers();
            }
        },
        updateTriggers: {
            getPosition: [appTime],
            getFillColor: [selectedConstellation, selectedSatId, searchTerm, hoveredSatId],
            getRadius: [selectedConstellation, selectedSatId, searchTerm, hoveredSatId]
        }
    });

    deckgl.setProps({ layers: [backgroundLayer, earthTextureLayer, satLayer] });
}

let lastTime = performance.now();
function startAnimation() {
    function tick(time) {
        const dt = (time - lastTime) / 1000.0;
        lastTime = time;
        
        // Orbital motion multiplier
        appTime += dt * 0.0025;
        
        // Camera tracking
        if (isTracking && trackedSatellite) {
            const [lon, lat] = calculatePosition(trackedSatellite, appTime);
            // Create a new viewState object, preserving zoom level set by user
            viewState = {
                ...viewState,
                longitude: lerpAngle(viewState.longitude, lon, 0.05),
                latitude: viewState.latitude + (lat - viewState.latitude) * 0.05
            };
            deckgl.setProps({viewState});
        }
        
        renderLayers();
        animationFrame = requestAnimationFrame(tick);
    }
    animationFrame = requestAnimationFrame(tick);
}

// UI Interactions
function selectSatellite(sat) {
    selectedSatId = sat.norad_id;
    trackedSatellite = sat;
    selectedConstellation = null; 
    isTracking = true; // Start camera tracking
    
    document.getElementById('sat-name').textContent = sat.name || "Unknown";
    document.getElementById('sat-id').textContent = sat.norad_id;
    document.getElementById('sat-constel').textContent = sat.satellite_constellation || "Unknown";
    document.getElementById('sat-country').textContent = sat.country || "Unknown";
    
    const r = sat.predicted_R;
    riskValue.textContent = r.toFixed(1);
    riskBar.style.width = `${r}%`;
    
    if (r < 50) riskValue.style.color = 'var(--text-cyan)';
    else if (r < 80) riskValue.style.color = '#f59e0b';
    else riskValue.style.color = 'var(--text-alert)';
    
    const ul = document.getElementById('shap-list');
    ul.innerHTML = '';
    if (sat.top_3_shap_features) {
        sat.top_3_shap_features.forEach(f => {
            const li = document.createElement('li');
            li.textContent = f;
            ul.appendChild(li);
        });
    }
    
    detailsPanel.classList.remove('hidden');
    renderLayers();
}

function clearSelection() {
    selectedSatId = null;
    trackedSatellite = null;
    selectedConstellation = null;
    isTracking = false;
    detailsPanel.classList.add('hidden');
    renderLayers();
}

closePanelBtn.addEventListener('click', () => {
    clearSelection();
});

highlightConstelBtn.addEventListener('click', () => {
    const constel = document.getElementById('sat-constel').textContent;
    if (constel && constel !== 'Other' && constel !== 'Unknown') {
        selectedConstellation = constel;
        renderLayers();
    }
});

searchInput.addEventListener('input', () => {
    renderLayers();
});

clearSearchBtn.addEventListener('click', () => {
    searchInput.value = '';
    renderLayers();
});
