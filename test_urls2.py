import urllib.request

urls = [
    "https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/planets/starfield_1024.png",
    "https://raw.githubusercontent.com/visgl/deck.gl-data/master/images/space.jpg",
    "https://svs.gsfc.nasa.gov/vis/a000000/a003800/a003895/starmap_2020.jpg",
    "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_4k.jpg"
]

for u in urls:
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        print(f"SUCCESS {res.getcode()}: {u}")
    except Exception as e:
        print(f"FAILED: {u} - {e}")
