import urllib.request

urls = [
    "https://raw.githubusercontent.com/visgl/deck.gl-data/master/images/earth.jpg",
    "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/earth.jpg",
    "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/globe/earth.jpg",
    "https://unpkg.com/@here/we-are-here@1.0.0/assets/earth.jpg",
    "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/earth_atmos_2048.jpg"
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        print(f"SUCCESS {res.getcode()}: {u}")
        break
    except Exception as e:
        print(f"FAILED: {u} - {e}")
