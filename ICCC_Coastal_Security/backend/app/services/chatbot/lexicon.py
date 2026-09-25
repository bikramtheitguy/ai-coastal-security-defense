"""Multilingual lexicon for the AI Maritime Public Assistant.

Languages: en, or (Odia), hi (Hindi), bn (Bengali), te (Telugu) + romanised
(transliterated) forms. Keyword lists are matched as substrings after
normalisation. THIS LEXICON IS A POC SEED AND MUST BE VALIDATED BY NATIVE
SPEAKERS AND COASTAL FIELD STAFF (dialect terms from Balasore, Kendrapara,
Puri and Ganjam fishing communities) before operational use.

Adding a language = add its script range in SCRIPTS, its keywords per concept
below, and its templates in responses.py. No code change is needed elsewhere.
"""
from __future__ import annotations

LANGS = {"en": "English", "or": "Odia", "hi": "Hindi", "bn": "Bengali", "te": "Telugu"}

# Unicode blocks used for script detection.
SCRIPTS = {
    "or": ("Odia", 0x0B00, 0x0B7F),
    "hi": ("Devanagari", 0x0900, 0x097F),
    "bn": ("Bengali", 0x0980, 0x09FF),
    "te": ("Telugu", 0x0C00, 0x0C7F),
}

# Native digits -> ASCII
DIGITS = {}
for base in (0x0B66, 0x0966, 0x09E6, 0x0C66):
    for i in range(10):
        DIGITS[chr(base + i)] = str(i)

# Romanised markers used to guess the language of Latin-script, non-English text.
ROMAN_MARKERS = {
    "or": ["danga", "donga", "achhi", "achi", "heba", "kana", "nahin", "nahi", "bhasi", "jauchi", "pani pasuchi",
           "kete jana", "mora", "amara", "amar danga", "chaluni", "kharap", "hoijaichi", "hela", "sathire"],
    "hi": ["hai", "hain", "nahi", "raha", "rahi", "gaya", "naav", "nav ", "hamari", "hamara", "mera", "kya", "kaha",
           "bandh ho", "beh rahi", "bah rahi", "log", "madad", "bachao", "jaldi"],
    "bn": ["amader", "amar", "nouka", "noukar", "bhese", "jachhe", "hoyeche", "bondho", "kothay", "sahajya", "bachao"],
    "te": ["padava", "padavalo", "ledu", "undi", "vundi", "aagipoyindi", "sahayam", "kapadandi", "manam", "memu"],
}

# Concept -> language -> keywords
C = {
    "BOAT": {
        "en": ["boat", "vessel", "trawler", "ship", "launch", "craft", "dinghy", "catamaran"],
        "or": ["ଡଙ୍ଗା", "ଟ୍ରଲର", "ଜାହାଜ", "ବୋଟ"],
        "hi": ["नाव", "नौका", "जहाज", "ट्रॉलर", "बोट"],
        "bn": ["নৌকা", "ট্রলার", "জাহাজ", "বোট"],
        "te": ["పడవ", "బోటు", "ట్రాలర్", "ఓడ", "బోట్"],
        "rom": ["danga", "donga", "naav", "nauka", "nouka", "padava", "bot "],
    },
    "PERSON": {
        "en": ["man ", "men ", "person", "people", "fisherman", "fishermen", "child", "boy", "girl", "father", "brother",
               "son ", "husband", "crew member", "tourist"],
        "or": ["ଲୋକ", "ମାଛୁଆ", "ପିଲା", "ଭାଇ", "ବାପା", "ପୁଅ", "ସ୍ୱାମୀ", "ଜଣ"],
        "hi": ["आदमी", "लोग", "मछुआरा", "मछुआरे", "बच्चा", "भाई", "पिता", "बेटा", "पति", "व्यक्ति"],
        "bn": ["মানুষ", "লোক", "জেলে", "বাচ্চা", "ভাই", "বাবা", "ছেলে", "স্বামী"],
        "te": ["మనిషి", "మనుషులు", "జాలరి", "మత్స్యకారు", "పిల్ల", "అన్న", "తమ్ముడు", "నాన్న", "కొడుకు", "భర్త"],
        "rom": ["loka", "machhua", "aadmi", "log ", "jele", "jaalari"],
    },
    "ENGINE": {
        "en": ["engine", "motor", "outboard"],
        "or": ["ଇଞ୍ଜିନ", "ମେସିନ", "ମୋଟର"],
        "hi": ["इंजन", "इंजिन", "मोटर", "मशीन"],
        "bn": ["ইঞ্জিন", "মোটর", "মেশিন"],
        "te": ["ఇంజన్", "ఇంజిన్", "మోటార్", "మెషిన్"],
        "rom": ["injin", "enjin", "mesin", "machine"],
    },
    "STOPPED": {
        "en": ["stopped", "stop", "not working", "failed", "failure", "broke", "dead", "won't start", "not starting",
               "breakdown", "broken"],
        "or": ["ବନ୍ଦ", "ଖରାପ", "କାମ କରୁନି", "ଚାଲୁନି", "ଚାଲୁ ହେଉନି"],
        "hi": ["बंद", "खराब", "काम नहीं", "रुक गया", "चल नहीं", "चालू नहीं"],
        "bn": ["বন্ধ", "খারাপ", "কাজ করছে না", "চলছে না", "থেমে"],
        "te": ["ఆగిపోయింది", "ఆగింది", "పని చేయడం లేదు", "పాడైంది", "చెడిపోయింది"],
        "rom": ["bandh", "band ho", "bondho", "kharap", "kharab", "chaluni", "aagipoyindi"],
    },
    "DRIFT": {
        "en": ["drifting", "drift", "adrift", "floating away", "carried by current", "pushed by current"],
        "or": ["ଭାସି", "ଭାସୁଛି"],
        "hi": ["बह रही", "बह रहा", "बहती", "बहाव", "भटक"],
        "bn": ["ভেসে", "ভাসছে"],
        "te": ["కొట్టుకుపో", "తేలుతూ", "కొట్టుకు"],
        "rom": ["bhasi", "beh rahi", "bah rahi", "beh raha", "bhese"],
    },
    "SINKING": {
        "en": ["sinking", "sink", "going down", "sank"],
        "or": ["ବୁଡ଼ୁଛି", "ବୁଡୁଛି", "ବୁଡ଼ି", "ବୁଡି"],
        "hi": ["डूब"],
        "bn": ["ডুবে", "ডুবছে"],
        "te": ["మునిగి", "మునుగుతో", "మునుగు"],
        "rom": ["buduchi", "budi jauchi", "doob", "dub raha", "dubche"],
    },
    "FLOODING": {
        "en": ["taking water", "water coming in", "water entering", "flooding", "leaking", "leak", "water inside"],
        "or": ["ପାଣି ପଶୁଛି", "ପାଣି ପଶି", "ପାଣି ଭରି"],
        "hi": ["पानी भर", "पानी आ रहा", "पानी घुस"],
        "bn": ["জল ঢুকছে", "পানি ঢুকছে", "জল ঢুক", "পানি ঢুক"],
        "te": ["నీరు వస్తోంది", "నీళ్ళు వస్తున్నాయి", "నీరు చేరుతోంది", "నీళ్లు వస్తున్నాయి"],
        "rom": ["pani pasuchi", "pani bhar", "pani aa raha", "jol dhukche"],
    },
    "CAPSIZE": {
        "en": ["capsized", "capsize", "overturned", "turned over", "flipped"],
        "or": ["ଓଲଟି"],
        "hi": ["पलट"],
        "bn": ["উল্টে", "উল্টে গেছে"],
        "te": ["బోల్తా", "తిరగబడ"],
        "rom": ["olati", "palat gayi", "palat gaya", "ulte gechhe"],
    },
    "OVERBOARD": {
        "en": ["man overboard", "fell into the sea", "fell overboard", "fell in the water", "fallen into sea",
               "fell into water"],
        "or": ["ସମୁଦ୍ରରେ ପଡ଼ି", "ପାଣିରେ ପଡ଼ି", "ସମୁଦ୍ରରେ ପଡି", "ପାଣିରେ ପଡି"],
        "hi": ["समुद्र में गिर", "पानी में गिर"],
        "bn": ["সমুদ্রে পড়ে", "জলে পড়ে", "পানিতে পড়ে"],
        "te": ["సముద్రంలో పడి", "నీటిలో పడి"],
        "rom": ["samudre padi", "pani me gir", "samundar me gir"],
    },
    "DROWNING": {
        "en": ["drowning", "drowned", "cannot swim", "can't swim"],
        "or": ["ବୁଡ଼ି ମରୁଛି", "ବୁଡ଼ିଯାଉଛି"],
        "hi": ["डूब रहा है", "डूब रही है", "डूब गया"],
        "bn": ["ডুবে যাচ্ছে", "ডুবে গেছে"],
        "te": ["మునిగిపోతున్నాడు", "మునిగిపోయాడు"],
        "rom": ["doob raha hai", "doob gaya"],
    },
    "FIRE": {
        "en": ["fire", "burning", "smoke", "flames"],
        "or": ["ନିଆଁ", "ଜଳୁଛି"],
        "hi": ["आग", "जल रहा", "जल रही", "धुआं", "धुआँ"],
        "bn": ["আগুন", "ধোঁয়া"],
        "te": ["మంటలు", "అగ్ని", "కాలిపోతోంది", "పొగ"],
        "rom": ["aag", "nian", "agun"],
    },
    "EXPLOSION": {
        "en": ["explosion", "blast", "exploded"],
        "or": ["ବିସ୍ଫୋରଣ"], "hi": ["धमाका", "विस्फोट"], "bn": ["বিস্ফোরণ"], "te": ["పేలుడు", "పేలింది"],
        "rom": ["dhamaka"],
    },
    "MEDICAL": {
        "en": ["injured", "injury", "bleeding", "unconscious", "heart attack", "chest pain", "sick", "fever", "medical",
               "hurt", "snake bite", "fracture", "not breathing", "vomiting", "fainted"],
        "or": ["ଆହତ", "ରକ୍ତ", "ଅଚେତ", "ବେମାର", "ଅସୁସ୍ଥ", "ଛାତି ଯନ୍ତ୍ରଣା"],
        "hi": ["घायल", "खून", "बेहोश", "बीमार", "चोट", "सीने में दर्द"],
        "bn": ["আহত", "রক্ত", "অজ্ঞান", "অসুস্থ", "বুকে ব্যথা"],
        "te": ["గాయ", "రక్తం", "స్పృహ", "అనారోగ్య", "జబ్బు", "ఛాతీ నొప్పి"],
        "rom": ["ahata", "ghayal", "behosh", "bimar", "osustho"],
    },
    "MEDICAL_SEVERE": {
        "en": ["unconscious", "not breathing", "heart attack", "heavy bleeding", "chest pain", "fainted"],
        "or": ["ଅଚେତ", "ଛାତି ଯନ୍ତ୍ରଣା"], "hi": ["बेहोश", "सीने में दर्द"], "bn": ["অজ্ঞান", "বুকে ব্যথা"],
        "te": ["స్పృహ", "ఛాతీ నొప్పి"], "rom": ["behosh"],
    },
    "COLLISION": {
        "en": ["collision", "collided", "hit by", "rammed", "crashed into", "hit another"],
        "or": ["ଧକ୍କା"], "hi": ["टक्कर", "टकरा"], "bn": ["ধাক্কা", "সংঘর্ষ"], "te": ["ఢీ"],
        "rom": ["dhakka", "takkar"],
    },
    "AGROUND": {
        "en": ["aground", "grounded", "stuck on sand", "sandbar", "sand bar", "mudflat", "mud flat", "stuck in mud",
               "stuck in sand"],
        "or": ["ବାଲିରେ ଫସି", "ଚଢ଼ାରେ", "କାଦୁଅରେ ଫସି", "ବାଲିଚର"],
        "hi": ["रेत में फंस", "कीचड़ में फंस", "रेत में फँस"],
        "bn": ["চরে আটকে", "বালিতে আটকে", "কাদায় আটকে"],
        "te": ["ఇసుకలో ఇరుక్కు", "బురదలో"],
        "rom": ["bali re phasi", "ret me phas"],
    },
    "TIDE": {
        "en": ["rising tide", "tide rising", "high tide", "tide is coming"],
        "or": ["ଜୁଆର"], "hi": ["ज्वार"], "bn": ["জোয়ার"], "te": ["పోటు"], "rom": ["juara", "jowar"],
    },
    "MISSING": {
        "en": ["missing", "not returned", "has not returned", "hasn't returned", "no contact", "lost contact",
               "not come back", "not came back", "did not return", "didn't return"],
        "or": ["ଫେରିନାହିଁ", "ଫେରି ନାହିଁ", "ନିଖୋଜ", "ଖବର ନାହିଁ", "ଫେରିନି"],
        "hi": ["लापता", "वापस नहीं", "नहीं लौट", "संपर्क नहीं"],
        "bn": ["নিখোঁজ", "ফেরেনি", "ফিরে আসেনি", "যোগাযোগ নেই"],
        "te": ["తప్పిపోయ", "తిరిగి రాలేదు", "ఆచూకీ", "సంబంధం లేదు"],
        "rom": ["lapata", "nikhoj", "pheri nahi", "pheruni", "wapas nahi", "nikhonj"],
    },
    "FUEL": {
        "en": ["out of fuel", "no fuel", "fuel finished", "fuel over", "diesel finished", "fuel shortage", "ran out of fuel"],
        "or": ["ତେଲ ସରି", "ଡିଜେଲ ସରି", "ତେଲ ନାହିଁ"], "hi": ["तेल खत्म", "डीजल खत्म", "ईंधन खत्म"],
        "bn": ["তেল শেষ", "ডিজেল শেষ"], "te": ["డీజిల్ అయిపోయింది", "ఇంధనం అయిపోయింది", "ఆయిల్ అయిపోయింది"],
        "rom": ["tel sari", "tel khatam", "diesel khatam"],
    },
    "STEERING": {
        "en": ["steering", "rudder"], "or": ["ଷ୍ଟିଅରିଂ", "ହାଲ"], "hi": ["स्टीयरिंग", "पतवार"],
        "bn": ["স্টিয়ারিং", "হাল"], "te": ["స్టీరింగ్", "చుక్కాని"], "rom": ["patwar"],
    },
    "PROPULSION": {
        "en": ["propeller", "propulsion", "shaft broken", "gearbox"], "or": ["ପଙ୍ଖା"], "hi": ["प्रोपेलर", "पंखा टूट"],
        "bn": ["প্রপেলার"], "te": ["ప్రొపెల్లర్"], "rom": ["pankha"],
    },
    "STORM": {
        "en": ["cyclone", "storm", "high waves", "rough sea", "big waves", "strong wind", "gale"],
        "or": ["ବାତ୍ୟା", "ଝଡ଼", "ଝଡ", "ବଡ଼ ଢେଉ", "ଢେଉ"],
        "hi": ["चक्रवात", "तूफान", "तूफ़ान", "ऊंची लहरें", "लहरें", "आंधी"],
        "bn": ["ঘূর্ণিঝড়", "ঝড়", "ঢেউ"],
        "te": ["తుఫాను", "తుపాను", "అలలు", "గాలివాన"],
        "rom": ["batya", "jhada", "toofan", "tufan", "jhor", "toofanu"],
    },
    "SUSPICIOUS": {
        "en": ["suspicious", "unknown boat", "strange boat", "unfamiliar boat", "foreign boat", "no name", "without lights",
               "no lights", "smuggling", "smuggle", "contraband", "infiltrat", "terrorist", "weapons", "suspect",
               "strangers"],
        "or": ["ସନ୍ଦେହ", "ଅଜଣା", "ଚୋରା ଚାଲାଣ", "ଚୋରାଚାଲାଣ"],
        "hi": ["संदिग्ध", "अनजान", "तस्करी", "अजनबी", "शक"],
        "bn": ["সন্দেহজনক", "অচেনা", "পাচার", "সন্দেহ"],
        "te": ["అనుమానాస్పద", "అనుమానం", "తెలియని", "స్మగ్లింగ్", "అక్రమ రవాణా"],
        "rom": ["sandeha", "sandigdh", "taskari", "ajana", "anjaan", "shak"],
    },
    "ALLEGATION": {  # words that express a CONCLUSION; recorded as allegation, never as fact
        "en": ["smuggling", "smuggler", "contraband", "terrorist", "infiltrat", "criminal", "drugs", "pirate"],
        "or": ["ଚୋରା ଚାଲାଣ", "ଚୋରାଚାଲାଣ", "ଆତଙ୍କବାଦୀ"], "hi": ["तस्करी", "तस्कर", "आतंकवादी"],
        "bn": ["পাচার", "সন্ত্রাসী"], "te": ["స్మగ్లింగ్", "అక్రమ రవాణా", "ఉగ్రవాది"],
        "rom": ["taskari", "taskar"],
    },
    "LANDING": {
        "en": ["landing", "landed", "came ashore", "unloading", "getting off", "coming ashore", "offloading"],
        "or": ["ଓହ୍ଲାଉଛନ୍ତି", "କୂଳରେ ଲାଗିଲା", "ଓହ୍ଲାଇଲେ"], "hi": ["उतर रहे", "उतार रहे", "किनारे पर उतर"],
        "bn": ["নামছে", "নামাচ্ছে", "তীরে নাম"], "te": ["దిగుతున్నారు", "ఒడ్డుకు", "దింపుతున్నారు"],
        "rom": ["utar rahe", "ohlauchhanti"],
    },
    "FOLLOWED": {
        "en": ["following us", "followed", "chasing", "being chased", "following our boat"],
        "or": ["ପିଛା", "ଗୋଡ଼ାଉଛି", "ଗୋଡାଉଛି"], "hi": ["पीछा"], "bn": ["পিছু", "তাড়া"], "te": ["వెంబడి"],
        "rom": ["picha", "peecha", "godauchi"],
    },
    "ATTACK": {
        "en": ["boarded", "hijack", "robbed", "robbery", "looted", "attacked", "threatened", "weapon", "gun", "knife",
               "kidnap"],
        "or": ["ଲୁଟ", "ଆକ୍ରମଣ", "ଅପହରଣ", "ବନ୍ଧୁକ"], "hi": ["लूट", "हमला", "अपहरण", "हथियार", "बंदूक"],
        "bn": ["ডাকাতি", "লুট", "হামলা", "অপহরণ", "বন্দুক"], "te": ["దోపిడీ", "దాడి", "కిడ్నాప్", "ఆయుధ", "తుపాకీ"],
        "rom": ["loot", "hamla", "apaharan"],
    },
    "HIJACK": {"en": ["hijack", "taken over our boat", "kidnap"], "or": ["ଅପହରଣ"], "hi": ["अपहरण"], "bn": ["অপহরণ"],
               "te": ["కిడ్నాప్", "హైజాక్"], "rom": ["apaharan"]},
    "ROBBERY": {"en": ["robbed", "robbery", "looted", "stole"], "or": ["ଲୁଟ", "ଚୋରି"], "hi": ["लूट", "चोरी"],
                "bn": ["ডাকাতি", "লুট", "চুরি"], "te": ["దోపిడీ", "దొంగ"], "rom": ["loot", "chori"]},
    "ILLEGAL_FISHING": {
        "en": ["illegal fishing", "banned net", "fishing ban", "turtle", "trawling near shore", "restricted fishing",
               "fishing in sanctuary"],
        "or": ["ବେଆଇନ ମାଛ", "କଇଁଛ", "ନିଷିଦ୍ଧ ଜାଲ"], "hi": ["अवैध मछली", "प्रतिबंधित जाल", "कछुआ"],
        "bn": ["অবৈধ মাছ", "নিষিদ্ধ জাল", "কচ্ছপ"], "te": ["అక్రమ చేపల", "నిషేధిత వల", "తాబేలు"],
        "rom": ["beaaina macha", "awaidh machhli"],
    },
    "POLLUTION": {
        "en": ["oil spill", "oil slick", "pollution", "dead fish", "chemical", "oil on water", "tar balls"],
        "or": ["ତେଲ ଛିଟିକି", "ପ୍ରଦୂଷଣ", "ମଲା ମାଛ", "ତେଲ ଭାସୁଛି"], "hi": ["तेल रिसाव", "प्रदूषण", "मरी मछली", "तेल फैल"],
        "bn": ["তেল ছড়িয়ে", "দূষণ", "মরা মাছ"], "te": ["చమురు", "కాలుష్యం", "చనిపోయిన చేపలు"],
        "rom": ["pradushan", "tel riswa"],
    },
    "OIL": {"en": ["oil spill", "oil slick", "oil on water", "tar balls"], "or": ["ତେଲ ଛିଟିକି", "ତେଲ ଭାସୁଛି"],
            "hi": ["तेल रिसाव", "तेल फैल"], "bn": ["তেল ছড়িয়ে"], "te": ["చమురు"], "rom": ["tel riswa"]},
    "OBSTRUCTION": {
        "en": ["floating object", "obstruction", "floating log", "container floating", "navigation hazard", "debris",
               "floating container", "ghost net", "buoy adrift", "wreck"],
        "or": ["ଭାସୁଥିବା ଜିନିଷ", "ଭଙ୍ଗା ଜାହାଜ"], "hi": ["तैरती वस्तु", "मलबा", "बहता कंटेनर"],
        "bn": ["ভাসমান বস্তু", "ধ্বংসাবশেষ"], "te": ["తేలియాడే వస్తువు", "శిథిలాలు"], "rom": ["malba"],
    },
    "ABANDONED": {
        "en": ["abandoned boat", "empty boat", "no one on board", "unmanned boat", "nobody on board", "abandoned vessel"],
        "or": ["ଖାଲି ଡଙ୍ଗା", "ଛାଡ଼ି ଦିଆଯାଇଥିବା"], "hi": ["खाली नाव", "लावारिस"], "bn": ["খালি নৌকা", "পরিত্যক্ত"],
        "te": ["ఖాళీ పడవ", "వదిలేసిన"], "rom": ["khali danga", "khali naav", "lawaris"],
    },
    "BEACH": {
        "en": ["beach", "swept away", "washed away", "while bathing", "swimming at", "sea beach"],
        "or": ["ବେଳାଭୂମି", "ଢେଉରେ ଭାସିଗଲା", "ଗାଧୋଇବା"], "hi": ["समुद्र तट", "बह गया", "नहाते समय"],
        "bn": ["সৈকত", "ভেসে গেছে", "স্নান করতে"], "te": ["బీచ్", "సముద్ర తీరం", "కొట్టుకుపోయాడు", "స్నానం"],
        "rom": ["belabhumi", "beh gaya", "bah gaya"],
    },
    "TOURIST": {
        "en": ["tourist boat", "jet ski", "parasail", "banana boat", "speed boat ride", "boat ride", "joy ride"],
        "or": ["ପର୍ଯ୍ୟଟକ ଡଙ୍ଗା"], "hi": ["पर्यटक नाव", "जेट स्की"], "bn": ["পর্যটক নৌকা"], "te": ["పర్యాటక పడవ"],
        "rom": [],
    },
    "RECREATIONAL": {"en": ["kayak", "sailing", "yacht", "surfing", "surfer", "canoe"], "or": [], "hi": [], "bn": [],
                     "te": [], "rom": []},
    "HARBOUR": {
        "en": ["harbour", "harbor", "jetty", "port", "landing centre", "landing center", "fish market"],
        "or": ["ଜେଟି", "ବନ୍ଦର", "ହାରବର"], "hi": ["बंदरगाह", "जेटी", "हार्बर"], "bn": ["জেটি", "হারবার", "বন্দর"],
        "te": ["ఓడరేవు", "హార్బర్", "జెట్టీ"], "rom": ["jetty", "bandar"],
    },
    "WEATHER": {
        "en": ["weather", "forecast", "wind speed", "rain", "is it safe to go", "can we go to sea", "sea condition",
               "warning"],
        "or": ["ପାଣିପାଗ", "ସତର୍କ", "ବର୍ଷା"], "hi": ["मौसम", "चेतावनी", "बारिश"], "bn": ["আবহাওয়া", "সতর্ক", "বৃষ্টি"],
        "te": ["వాతావరణం", "హెచ్చరిక", "వర్షం"], "rom": ["mausam", "panipaga", "abhawa"],
    },
    "STATION": {
        "en": ["nearest police", "police station", "marine police", "contact police", "phone number", "nearest station"],
        "or": ["ଥାନା", "ପୋଲିସ"], "hi": ["थाना", "पुलिस स्टेशन", "नजदीकी पुलिस"], "bn": ["থানা", "পুলিশ"],
        "te": ["పోలీస్ స్టేషన్", "పోలీసు"], "rom": ["thana"],
    },
    "SAFETY": {
        "en": ["safety", "life jacket", "lifejacket", "what should i carry", "safety tips", "guidance", "precaution"],
        "or": ["ସୁରକ୍ଷା", "ଲାଇଫ ଜ୍ୟାକେଟ", "ଲାଇଫ୍ ଜ୍ୟାକେଟ୍"], "hi": ["सुरक्षा", "लाइफ जैकेट"],
        "bn": ["নিরাপত্তা", "লাইফ জ্যাকেট"], "te": ["భద్రత", "లైఫ్ జాకెట్"], "rom": ["suraksha"],
    },
    "VHF": {"en": ["vhf", "radio not working", "radio failed", "wireless set", "radio set"], "or": ["ରେଡିଓ"],
            "hi": ["रेडियो"], "bn": ["রেডিও"], "te": ["రేడియో"], "rom": ["radio"]},
    "LOCATION_HELP": {
        "en": ["how to share location", "share location", "send location", "live location", "how to send my location"],
        "or": ["ଲୋକେସନ୍ କିପରି", "ଲୋକେସନ କିପରି"], "hi": ["लोकेशन कैसे"], "bn": ["লোকেশন কিভাবে"],
        "te": ["లొకేషన్ ఎలా"], "rom": ["location kaise"],
    },
    "HUMAN": {"en": ["operator", "human", "talk to police", "speak to someone", "call me"],
              "or": ["ଅପରେଟର"], "hi": ["ऑपरेटर", "इंसान से बात"], "bn": ["অপারেটর"], "te": ["ఆపరేటర్"], "rom": []},
    "URGENT": {"en": ["help", "urgent", "emergency", "save us", "sos", "mayday", "quickly", "hurry"],
               "or": ["ସାହାଯ୍ୟ", "ବଞ୍ଚାଅ", "ଜରୁରୀ", "ଶୀଘ୍ର"], "hi": ["मदद", "बचाओ", "जल्दी", "तुरंत"],
               "bn": ["সাহায্য", "বাঁচাও", "জরুরি", "তাড়াতাড়ি"], "te": ["సహాయం", "కాపాడండి", "అత్యవసరం", "తొందరగా"],
               "rom": ["bachao", "madad", "bachaa", "sahajya", "kapadandi"]},
    "LIFEJACKETS_YES": {"en": ["wearing life jacket", "have life jackets", "life jackets on", "wearing lifejacket"],
                        "or": ["ଲାଇଫ୍ ଜ୍ୟାକେଟ୍ ପିନ୍ଧିଛୁ", "ଜ୍ୟାକେଟ ପିନ୍ଧିଛୁ"], "hi": ["जैकेट पहन"],
                        "bn": ["জ্যাকেট পরে"], "te": ["జాకెట్ వేసుకున్నాం"], "rom": []},
    "NO_INJURY": {"en": ["no one injured", "nobody injured", "no injury", "no injuries", "all are fine", "everyone is fine",
                         "all safe", "no one is hurt"],
                  "or": ["କେହି ଆହତ ନାହାଁନ୍ତି", "ସମସ୍ତେ ଭଲ", "ସମସ୍ତେ ଠିକ୍", "ସମସ୍ତେ ଠିକ", "ଆହତ ନାହାନ୍ତି"],
                  "hi": ["कोई घायल नहीं", "सब ठीक", "सभी ठीक"], "bn": ["কেউ আহত নয়", "সবাই ভালো", "সবাই ঠিক"],
                  "te": ["ఎవరూ గాయపడలేదు", "అందరూ బాగున్నారు", "అందరూ క్షేమం"], "rom": ["sabu thik", "sab theek"]},
    "NO_WATER": {"en": ["no water coming", "not taking water", "no water inside", "no leak"],
                 "or": ["ପାଣି ପଶୁନି", "ପାଣି ପଶୁ ନାହିଁ"], "hi": ["पानी नहीं आ रहा", "पानी नहीं भर"],
                 "bn": ["জল ঢুকছে না", "পানি ঢুকছে না"], "te": ["నీరు రావడం లేదు"], "rom": []},
}

YES = {"en": ["yes", "yeah", "yep", "haan"], "or": ["ହଁ", "ହଁ।", "ହଉ"], "hi": ["हाँ", "हां", "जी"], "bn": ["হ্যাঁ", "হাঁ"],
       "te": ["అవును", "ఔను"], "rom": ["han", "haan", "ha "]}
NO = {"en": ["no", "none", "nobody", "not"], "or": ["ନାହିଁ", "ନା", "ନାହାଁନ୍ତି"], "hi": ["नहीं", "ना"],
      "bn": ["না", "নেই"], "te": ["లేదు", "కాదు", "లేరు"], "rom": ["nahi", "na "]}

# Number words (people counts) per language
NUMBER_WORDS = {
    "en": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20},
    "or": {"ଗୋଟିଏ": 1, "ଜଣେ": 1, "ଦୁଇ": 2, "ଦୁଇଜଣ": 2, "ତିନି": 3, "ଚାରି": 4, "ପାଞ୍ଚ": 5, "ଛଅ": 6, "ସାତ": 7,
           "ଆଠ": 8, "ନଅ": 9, "ଦଶ": 10},
    "hi": {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8,
           "नौ": 9, "दस": 10},
    "bn": {"এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "দশ": 10},
    "te": {"ఒకరు": 1, "ఒక": 1, "ఇద్దరు": 2, "ముగ్గురు": 3, "నలుగురు": 4, "ఐదుగురు": 5, "ఆరుగురు": 6,
           "ఏడుగురు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10},
}

# Classifier words that follow a number when counting people ("5 ଜଣ", "5 लोग", "5 জন", "5 మంది")
PERSON_COUNTERS = ["people", "persons", "person", "men", "crew", "fishermen", "of us", "onboard", "on board",
                   "ଜଣ", "ଲୋକ", "लोग", "आदमी", "व्यक्ति", "जन", "জন", "লোক", "మంది", "మనుషులు", "jana", "log", "jon", "mandi"]

# Gloss used to build the word-level canonical English interpretation.
GLOSS = {
    "ENGINE": "engine", "STOPPED": "stopped/failed", "DRIFT": "drifting", "SINKING": "sinking",
    "FLOODING": "taking water", "CAPSIZE": "capsized", "OVERBOARD": "person fell into sea", "DROWNING": "drowning",
    "FIRE": "fire", "EXPLOSION": "explosion", "MEDICAL": "injury/medical", "COLLISION": "collision",
    "AGROUND": "aground/stuck", "TIDE": "tide", "MISSING": "missing/not returned", "FUEL": "out of fuel",
    "STEERING": "steering", "PROPULSION": "propeller/propulsion", "STORM": "storm/rough sea",
    "SUSPICIOUS": "suspicious/unknown", "ALLEGATION": "(citizen allegation)", "LANDING": "landing/unloading ashore",
    "FOLLOWED": "being followed", "ATTACK": "attack/threat", "ILLEGAL_FISHING": "illegal fishing",
    "POLLUTION": "pollution", "OBSTRUCTION": "floating obstruction", "ABANDONED": "abandoned/empty boat",
    "BEACH": "beach", "HARBOUR": "harbour/jetty", "WEATHER": "weather", "STATION": "police station",
    "SAFETY": "safety", "VHF": "radio/VHF", "BOAT": "boat", "PERSON": "person(s)", "URGENT": "help/urgent",
    "HUMAN": "wants human operator", "NO_INJURY": "no injuries", "NO_WATER": "no water ingress",
    "LIFEJACKETS_YES": "wearing life jackets", "LOCATION_HELP": "how to share location", "TOURIST": "tourist craft",
    "RECREATIONAL": "recreational craft", "HIJACK": "hijack", "ROBBERY": "robbery", "OIL": "oil spill",
    "MEDICAL_SEVERE": "severe medical",
}
