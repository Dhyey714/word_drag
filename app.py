from flask import Flask, render_template, jsonify, request
import random
import time
import nltk

nltk.download('words',   quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import words as nltk_words
from nltk.corpus import wordnet

print("Loading dictionary...")
BASE_WORDS = set(
    w.lower() for w in nltk_words.words()
    if 3 <= len(w) <= 8 and w.isalpha()
)

# Curated common modern words NLTK misses
EXTRA_WORDS = {
    # money / finance
    "paid","pays","payer","payers","paying","repaid","prepaid",
    "cash","cashed","cashes","fund","funds","funded","loan","loans","loaned",
    "debt","debts","earn","earns","earned","wage","wages","cost","costs","costed",
    # gaming
    "gamer","gamers","gaming","gamed","loot","loots","looted","looting",
    "grind","grinds","grinded","grinding","spawn","spawns","spawned",
    "noob","noobs","boss","bosses","quest","quests","level","levels","leveled",
    "rank","ranks","ranked","ranking","clan","clans","guild","guilds",
    "raid","raids","raided","raiding","lobby","lobbies","ping","pings","lag","lags",
    "meta","nerf","nerfs","nerfed","buff","buffs","buffed","skin","skins",
    "item","items","slot","slots","drop","drops","dropped",
    # social / tech
    "emoji","emojis","selfie","selfies","blog","blogs","blogger","blogged",
    "vlog","vlogs","vlogger","vlogged","app","apps","wifi","pixel","pixels",
    "tweet","tweets","tweeted","meme","memes","viral","vibes","vibe","vibed",
    "stan","stans","hype","hyped","fan","fans","streamer","streamers",
    "stream","streams","streamed","chat","chats","chatted","reply","replies",
    "liked","likes","follow","follows","followed","follower","followers",
    "post","posts","posted","share","shares","shared",
    "hack","hacks","hacked","hacker","code","coded","coder","codes",
    "byte","bytes","data","crypto","token","tokens","node","nodes",
    "reel","reels","filter","filters","filtered","edit","edits","edited",
    # everyday
    "okay","okays","sure","nope","yep","yeah","nah","bruh","dude","dudes",
    "chill","chills","chilled","flex","flexed","slay","slays","slayed",
    "queen","queens","icon","icons","lit","goat","goats","mid","sus","ratio",
    "rant","rants","ranted","gripe","gripes","griped",
    "trek","treks","trekked","hike","hikes","hiked","hiker","hikers",
    "camp","camps","camped","camping","camper","campers",
    "bike","bikes","biked","biker","bikers","surf","surfs","surfed","surfer",
    "skate","skates","skated","skater","snap","snaps","snapped",
    "zoom","zooms","zoomed","blend","blends","blended",
    # common short words often missing
    "via","ops","pro","pros","con","cons","hub","hubs","tab","tabs",
    "bid","bids","bidden","bit","bits","bog","bogs","bogged",
    "boa","boas","bode","bodes","bod","bods",
}

WORDS = BASE_WORDS | EXTRA_WORDS
print(f"Dictionary loaded: {len(WORDS)} words")

app = Flask(__name__)

# Scrabble-weighted letter pool (~42% vowels)
LETTER_POOL = (
    "AAAAAAAAA" "BB" "CC" "DDDD" "EEEEEEEEEEEE" "FF" "GGG" "HH"
    "IIIIIIIII" "J" "K" "LLLL" "MM" "NNNNNN" "OOOOOOOO" "PP" "Q"
    "RRRRRR" "SSSS" "TTTTTT" "UUUU" "VV" "WW" "X" "YY" "Z"
)
VOWELS = set("AEIOU")


def generate_grid(size=4):
    total = size * size
    min_v = max(3, int(total * 0.36))
    for _ in range(200):
        flat = [random.choice(LETTER_POOL) for _ in range(total)]
        if sum(1 for c in flat if c in VOWELS) >= min_v:
            return [flat[r * size:(r + 1) * size] for r in range(size)]
    flat = [random.choice(LETTER_POOL) for _ in range(total)]
    for i in random.sample(range(total), min_v):
        flat[i] = random.choice("AAEEEIIOOU")
    return [flat[r * size:(r + 1) * size] for r in range(size)]


def is_valid_word(word, min_len):
    w = word.lower()
    if len(w) < min_len:
        return False
    if w in WORDS:
        return True
    # -s / -es plurals and 3rd-person
    if w.endswith("s") and len(w) > 3 and w[:-1] in WORDS:
        return True
    if w.endswith("es") and len(w) > 4 and w[:-2] in WORDS:
        return True
    # -ed past tense (drop e or not)
    if w.endswith("ed") and len(w) > 4 and w[:-2] in WORDS:
        return True
    if w.endswith("ed") and len(w) > 4 and (w[:-1]) in WORDS:
        return True
    # -ing (drop e or not)
    if w.endswith("ing") and len(w) > 5 and w[:-3] in WORDS:
        return True
    if w.endswith("ing") and len(w) > 5 and (w[:-3] + "e") in WORDS:
        return True
    # -er / -ers agent nouns
    if w.endswith("er") and len(w) > 4 and w[:-2] in WORDS:
        return True
    if w.endswith("ers") and len(w) > 5 and w[:-3] in WORDS:
        return True
    return False


def get_neighbors(r, c, size):
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
        and 0 <= r + dr < size and 0 <= c + dc < size
    ]


def _score_for_best(w):
    """
    Score used to pick the 'best' (most recognizable) word.
    Prefer words from EXTRA_WORDS (curated/common), then prefer
    medium length (4-6 letters). Heavily penalise very long or
    very short; never pick something >6 letters unless nothing
    else is available.
    """
    extra = 500 if w in EXTRA_WORDS else 0
    base  = w in BASE_WORDS and w not in EXTRA_WORDS
    length = len(w)
    # sweet-spot length bonus
    len_score = {3: 10, 4: 40, 5: 60, 6: 50}.get(length, 0)
    # penalise obscure long words from NLTK corpus
    if length >= 7 and not extra:
        len_score -= 200
    return extra + len_score


def find_best_word(grid, min_len=3):
    """Return (WORD, path) — the most recognisable valid word in grid."""
    size = len(grid)
    found = {}   # lowercase word → path

    def dfs(r, c, visited, cur, path):
        if len(cur) >= min_len and is_valid_word(cur, min_len):
            w = cur.lower()
            if w not in found:
                found[w] = list(path)
        if len(cur) >= 7:
            return
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                dfs(nr, nc, visited, cur + grid[nr][nc], path + [[nr, nc]])
                visited.remove((nr, nc))

    for r in range(size):
        for c in range(size):
            dfs(r, c, {(r, c)}, grid[r][c], [[r, c]])

    if not found:
        return "", []
    best = max(found, key=_score_for_best)
    return best.upper(), found[best]


def find_all_words(grid, min_len=3, time_limit=10.0):
    """Return dict {lowercase_word: path} for all valid words in the grid."""
    size  = len(grid)
    found = {}
    t0    = time.time()

    def dfs(r, c, visited, cur, path):
        if time.time() - t0 > time_limit:
            return
        if len(cur) >= min_len and is_valid_word(cur, min_len):
            w = cur.lower()
            if w not in found:
                found[w] = list(path)
        if len(cur) >= 8:
            return
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                dfs(nr, nc, visited, cur + grid[nr][nc], path + [[nr, nc]])
                visited.remove((nr, nc))

    for r in range(size):
        for c in range(size):
            if time.time() - t0 > time_limit:
                break
            dfs(r, c, {(r, c)}, grid[r][c], [[r, c]])

    return found


def get_definition(word):
    """Return (definition_string, pos_string) from WordNet, or (None, None)."""
    synsets = wordnet.synsets(word.lower())
    if not synsets:
        return None, None
    s   = synsets[0]
    pos = {"n": "noun", "v": "verb", "a": "adjective",
           "s": "adjective", "r": "adverb"}.get(s.pos(), "")
    return s.definition(), pos


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new-grid")
def api_new_grid():
    size = max(3, min(6, int(request.args.get("size", 4))))
    return jsonify({"grid": generate_grid(size), "size": size})


@app.route("/api/check-word", methods=["POST"])
def api_check_word():
    data    = request.get_json()
    word    = data.get("word", "").strip().lower()
    min_len = int(data.get("min_len", 3))
    valid   = is_valid_word(word, min_len)
    pts     = 0
    if valid:
        pts = {3: 100, 4: 200, 5: 400, 6: 700, 7: 1200}.get(len(word), 2000)
    return jsonify({"valid": valid, "score": pts, "word": word.upper()})


@app.route("/api/best-word", methods=["POST"])
def api_best_word():
    data    = request.get_json()
    grid    = data.get("grid", [])
    min_len = int(data.get("min_len", 3))
    if not grid:
        return jsonify({"word": "", "path": []})
    word, path = find_best_word(grid, min_len)
    return jsonify({"word": word, "path": path})


@app.route("/api/all-words", methods=["POST"])
def api_all_words():
    data    = request.get_json()
    grid    = data.get("grid", [])
    min_len = int(data.get("min_len", 3))
    if not grid:
        return jsonify({"words": [], "truncated": False})
    t0    = time.time()
    found = find_all_words(grid, min_len, time_limit=10.0)
    truncated = (time.time() - t0) >= 9.9
    words_list = sorted(
        [{"word": w.upper(), "path": p} for w, p in found.items()],
        key=lambda x: (-len(x["word"]), x["word"])
    )
    return jsonify({"words": words_list, "truncated": truncated})


@app.route("/api/hint", methods=["POST"])
def api_hint():
    data    = request.get_json()
    grid    = data.get("grid", [])
    min_len = int(data.get("min_len", 3))
    found   = set(w.lower() for w in data.get("found", []))
    if not grid:
        return jsonify({"word": "", "path": []})

    size       = len(grid)
    candidates = {}

    def dfs(r, c, visited, cur, path):
        if len(cur) == min_len:
            if is_valid_word(cur, min_len) and cur.lower() not in found:
                candidates[cur.lower()] = list(path)
            return
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                dfs(nr, nc, visited, cur + grid[nr][nc], path + [[nr, nc]])
                visited.remove((nr, nc))

    for r in range(size):
        for c in range(size):
            dfs(r, c, {(r, c)}, grid[r][c], [[r, c]])

    if not candidates:
        return jsonify({"word": "", "path": []})

    items = list(candidates.items())
    random.shuffle(items)
    w, p = items[0]
    return jsonify({"word": w.upper(), "path": p})


@app.route("/api/definitions", methods=["POST"])
def api_definitions():
    data  = request.get_json()
    words = data.get("words", [])
    result = []
    for w in words:
        defn, pos = get_definition(w)
        result.append({"word": w, "definition": defn, "pos": pos})
    return jsonify({"definitions": result})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
