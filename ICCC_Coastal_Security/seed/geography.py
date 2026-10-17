"""Synthetic geography for the POC.

IMPORTANT - SIMULATED / POC DATA:
  * The six coastal district names are public (Balasore, Bhadrak, Kendrapara,
    Jagatsinghpur, Puri, Ganjam). Their coordinates here are approximate centroids.
  * The 18 Marine Police Station names below are ILLUSTRATIVE coastal localities chosen
    for the demonstration. They are NOT asserted to be the official list or official
    locations of Odisha Marine Police Stations. Replace with authoritative master data.
  * Coordinates are approximate and hand-placed; they are not survey or official data.
  * The coastline is a coarse, hand-digitised APPROXIMATION for the offline map fallback
    only. It is not an official boundary and must not be used for navigation.
  * "Sensitive installations", vulnerable landing points, radar/EO sites and restricted
    zones are generic placeholders at arbitrary positions; they do not represent real sites.
"""

DISTRICTS = [
    ("BLS", "Balasore", 21.45, 87.00),
    ("BDK", "Bhadrak", 20.95, 86.80),
    ("KDP", "Kendrapara", 20.55, 86.70),
    ("JSP", "Jagatsinghpur", 20.20, 86.45),
    ("PRI", "Puri", 19.85, 85.85),
    ("GJM", "Ganjam", 19.30, 84.95),
]

# code, name, district, lat, lon, native-script aliases (for chatbot place matching)
STATIONS = [
    ("MPS-TLS", "Talasari", "BLS", 21.585, 87.470, ["ତାଳସାରି", "तालसारी", "তালসারি"]),
    ("MPS-CDP", "Chandipur", "BLS", 21.447, 87.025, ["ଚାନ୍ଦିପୁର", "चांदीपुर", "চাঁদিপুর", "చాందీపూర్"]),
    ("MPS-BHB", "Bahabalpur", "BLS", 21.300, 86.935, ["ବାହାବଳପୁର"]),
    ("MPS-CDB", "Chandbali", "BDK", 20.780, 86.745, ["ଚାନ୍ଦବାଲି", "चांदबाली"]),
    ("MPS-DHM", "Dhamra", "BDK", 20.795, 86.960, ["ଧାମରା", "धामरा", "ধামরা", "ధామ్రా"]),
    ("MPS-CHD", "Chudamani", "BDK", 21.000, 86.830, ["ଚୂଡ଼ାମଣି", "चूड़ामणि"]),
    ("MPS-TLC", "Talchua", "KDP", 20.665, 86.905, ["ତାଳଚୁଆ"]),
    ("MPS-JMB", "Jambu", "KDP", 20.555, 86.790, ["ଜମ୍ବୁ", "जम्बू"]),
    ("MPS-KHN", "Kharinashi", "KDP", 20.445, 86.725, ["ଖରିନାଶି"]),
    ("MPS-PDP", "Paradip", "JSP", 20.265, 86.675, ["ପାରାଦୀପ", "पारादीप", "পারাদ্বীপ", "పారాదీప్", "paradeep"]),
    ("MPS-SIA", "Siali", "JSP", 20.150, 86.500, ["ସିଆଳି"]),
    ("MPS-AST", "Astaranga", "PRI", 19.985, 86.340, ["ଅସ୍ତରଙ୍ଗ", "अस्तरंग", "আস্তরঙ্গ"]),
    ("MPS-KNK", "Konark", "PRI", 19.880, 86.105, ["କୋଣାର୍କ", "कोणार्क", "কোনারক", "కోణార్క్"]),
    ("MPS-PUR", "Puri", "PRI", 19.795, 85.835, ["ପୁରୀ", "पुरी", "পুরী", "పూరీ"]),
    ("MPS-STP", "Satapada", "PRI", 19.670, 85.455, ["ସାତପଡ଼ା", "सातपड़ा"]),
    ("MPS-ARJ", "Arjyapalli", "GJM", 19.380, 85.055, ["ଆର୍ଯ୍ୟପଲ୍ଲୀ"]),
    ("MPS-GPL", "Gopalpur", "GJM", 19.260, 84.915, ["ଗୋପାଳପୁର", "गोपालपुर", "গোপালপুর", "గోపాల్‌పూర్"]),
    ("MPS-SNP", "Sonapur", "GJM", 19.110, 84.790, ["ସୋନପୁର"]),
]

# Seaward bearing (deg) per district: the Odisha coast trends SW-NE, the sea lies to the E/SE.
SEAWARD = {"BLS": 120, "BDK": 105, "KDP": 110, "JSP": 125, "PRI": 145, "GJM": 140}

# Coarse approximate coastline, south-west to north-east ([lon, lat]). NOT OFFICIAL.
COASTLINE = [
    [84.70, 19.00], [84.79, 19.10], [84.915, 19.25], [85.055, 19.37], [85.25, 19.50], [85.45, 19.66],
    [85.65, 19.73], [85.835, 19.79], [86.00, 19.84], [86.105, 19.875], [86.25, 19.93], [86.34, 19.98],
    [86.45, 20.08], [86.52, 20.15], [86.62, 20.22], [86.68, 20.26], [86.73, 20.38], [86.75, 20.45],
    [86.80, 20.55], [86.88, 20.62], [86.92, 20.67], [86.97, 20.76], [86.97, 20.82], [86.88, 20.93],
    [86.84, 21.00], [86.90, 21.15], [86.94, 21.30], [87.02, 21.44], [87.20, 21.52], [87.47, 21.58],
    [87.55, 21.62],
]

# Other named reference places (public geographic names; positions approximate).
PORTS = [
    ("PORT-PDP", "Paradip Port", "PORT", "JSP", 20.262, 86.690, ["ପାରାଦୀପ ବନ୍ଦର", "पारादीप बंदरगाह"]),
    ("PORT-DHM", "Dhamra Port", "PORT", "BDK", 20.830, 87.020, ["ଧାମରା ବନ୍ଦର"]),
    ("PORT-GPL", "Gopalpur Port", "PORT", "GJM", 19.290, 84.955, ["ଗୋପାଳପୁର ବନ୍ଦର"]),
    ("FH-PDP", "Paradip Fishing Harbour", "FISHING_HARBOUR", "JSP", 20.285, 86.665, []),
    ("FH-DHM", "Dhamra Fishing Harbour", "FISHING_HARBOUR", "BDK", 20.800, 86.945, []),
    ("FH-BHB", "Bahabalpur Fishing Harbour", "FISHING_HARBOUR", "BLS", 21.310, 86.925, []),
    ("FH-ARJ", "Arjyapalli Fishing Harbour", "FISHING_HARBOUR", "GJM", 19.395, 85.045, []),
    ("ISL-SHT", "Short Island (approx.)", "ISLAND", "BDK", 20.900, 87.030, []),
    ("ISL-HKT", "Hukitola Island (approx.)", "ISLAND", "JSP", 20.330, 86.760, []),
    ("ISL-NLB", "Nalabana Island, Chilika (approx.)", "ISLAND", "PRI", 19.740, 85.320, []),
    ("RM-SUB", "Subarnarekha river mouth", "RIVER_MOUTH", "BLS", 21.570, 87.400, []),
    ("RM-BDB", "Budhabalanga river mouth", "RIVER_MOUTH", "BLS", 21.470, 87.060, []),
    ("RM-DHM", "Dhamra river mouth", "RIVER_MOUTH", "BDK", 20.790, 86.990, []),
    ("RM-MHN", "Mahanadi river mouth", "RIVER_MOUTH", "JSP", 20.310, 86.740, ["ମହାନଦୀ ମୁହାଣ"]),
    ("RM-DEV", "Devi river mouth", "RIVER_MOUTH", "PRI", 19.960, 86.390, []),
    ("RM-RSK", "Rushikulya river mouth", "RIVER_MOUTH", "GJM", 19.370, 85.070, []),
    ("EST-CHL", "Chilika lagoon mouth", "ESTUARY", "PRI", 19.690, 85.470, ["ଚିଲିକା", "चिल्का"]),
    ("EST-BTR", "Brahmani-Baitarani estuary", "ESTUARY", "KDP", 20.700, 86.960, []),
    ("CRK-01", "Creek C-1 (SIMULATED)", "CREEK", "KDP", 20.600, 86.880, []),
    ("CRK-02", "Creek C-2 (SIMULATED)", "CREEK", "BLS", 21.250, 86.950, []),
    ("CRK-03", "Creek C-3 (SIMULATED)", "CREEK", "PRI", 19.930, 86.230, []),
]
