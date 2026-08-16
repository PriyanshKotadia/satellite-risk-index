// mock_deck.js
let viewState = { zoom: 0 };
let isTracking = false; // Zoom broken when not tracking!

// Deck.gl internal state (simplified)
let internalState = { viewState: { zoom: 0 } };

function setProps(props) {
    if (props.viewState) {
        internalState.viewState = props.viewState;
        console.log("DeckGL received new viewState prop. Zoom is now:", internalState.viewState.zoom);
    } else {
        // DeckGL merges props. If viewState is omitted, it keeps its old viewState!
    }
}

function renderLayers() {
    setProps({ layers: [] }); // OMITTED viewState!
}

function onViewStateChange(newViewState, interactionState) {
    console.log("User scrolled! onViewStateChange fired with zoom:", newViewState.zoom);
    viewState = newViewState;
    renderLayers(); // Called without passing viewState to deck.gl!
}

// Simulate user zooming in
console.log("--- Initial State ---");
console.log("Internal Deck.gl Zoom:", internalState.viewState.zoom);

console.log("\n--- User scrolls mouse wheel ---");
// User scrolls, DeckGL computes new zoom = 1 based on its internal state
let newZoom = internalState.viewState.zoom + 1;
onViewStateChange({ zoom: newZoom }, { isZooming: true });

console.log("\n--- After scroll event ---");
console.log("App variable 'viewState' zoom:", viewState.zoom);
console.log("Internal Deck.gl Zoom (what actually renders!):", internalState.viewState.zoom);
console.log("\nCONCLUSION: Because renderLayers() doesn't pass viewState back to setProps, the zoom is lost when isTracking is false!");
