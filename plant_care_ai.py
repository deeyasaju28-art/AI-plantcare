#!/usr/bin/env python3
"""
==========================================================
  🌿 AI PLANT CARE ASSISTANT  (Terminal Application)
==========================================================
Features
  1. Plant knowledge base (15 popular plants)
  2. Soil-moisture simulation model (temp, humidity, light, pot, season)
  3. Machine Learning (Decision Tree) -> predicts: WATER NOW / WAIT / OVERWATER RISK
  4. Symptom diagnosis engine (Naive-Bayes style probability scoring)
  5. Plant Health Score (0-100) + personalised care tips
  6. Graph dashboard saved as PNG (soil forecast, environment match,
     diagnosis probabilities, ML feature importance)

Run:      python3 plant_care_ai.py
Requires: pip install numpy matplotlib scikit-learn
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")              # works in terminal / no display needed
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# 1. KNOWLEDGE BASE
#    water_days = normal interval (at 24C, 50% humidity)
#    threshold  = soil moisture % at which the plant needs water
# ----------------------------------------------------------------------
PLANTS = {
    "snake plant":  dict(water_days=14, threshold=20, light="low",    temp=(15, 30), hum=(30, 50), pet_safe=False, tip="Very hard to kill. Let soil dry fully; never leave it in standing water."),
    "aloe vera":    dict(water_days=14, threshold=20, light="high",   temp=(13, 32), hum=(30, 50), pet_safe=False, tip="Water deeply but rarely. Needs bright light and gritty, fast-draining soil."),
    "cactus":       dict(water_days=21, threshold=15, light="high",   temp=(10, 35), hum=(20, 40), pet_safe=True,  tip="Water only when soil is bone dry. Reduce watering sharply in winter."),
    "jade plant":   dict(water_days=14, threshold=25, light="high",   temp=(10, 30), hum=(30, 50), pet_safe=False, tip="Succulent leaves store water - wrinkled leaves mean thirsty."),
    "pothos":       dict(water_days=7,  threshold=35, light="medium", temp=(15, 30), hum=(40, 60), pet_safe=False, tip="Forgiving vine. Droopy leaves = thirsty. Trim to encourage bushiness."),
    "money plant":  dict(water_days=7,  threshold=35, light="medium", temp=(15, 30), hum=(40, 60), pet_safe=False, tip="Grows in soil or water. Yellow leaves usually mean overwatering."),
    "spider plant": dict(water_days=7,  threshold=40, light="medium", temp=(13, 27), hum=(40, 60), pet_safe=True,  tip="Brown tips = fluoride in tap water or dry air. Use filtered/rain water."),
    "peace lily":   dict(water_days=6,  threshold=45, light="low",    temp=(18, 30), hum=(50, 70), pet_safe=False, tip="Dramatically droops when thirsty, then recovers. Loves humidity."),
    "rubber plant": dict(water_days=8,  threshold=35, light="medium", temp=(16, 29), hum=(40, 60), pet_safe=False, tip="Wipe leaves monthly so they can photosynthesise well."),
    "fiddle leaf fig": dict(water_days=8, threshold=40, light="high", temp=(16, 29), hum=(40, 65), pet_safe=False, tip="Dislikes being moved. Needs bright indirect light and consistent watering."),
    "boston fern":  dict(water_days=4,  threshold=55, light="medium", temp=(16, 27), hum=(60, 80), pet_safe=True,  tip="Keep soil evenly moist. Mist often or place on a pebble tray."),
    "tulsi":        dict(water_days=3,  threshold=45, light="high",   temp=(20, 35), hum=(40, 60), pet_safe=True,  tip="Loves sun and regular water. Pinch flower buds for more leaves."),
    "orchid":       dict(water_days=7,  threshold=40, light="medium", temp=(18, 29), hum=(50, 70), pet_safe=True,  tip="Water roots, not the crown. Use bark mix and good drainage."),
    "monstera":     dict(water_days=7,  threshold=35, light="medium", temp=(18, 30), hum=(50, 70), pet_safe=False, tip="Give a moss pole to climb. Large leaf splits show a happy plant."),
    "hibiscus":     dict(water_days=3,  threshold=45, light="high",   temp=(18, 35), hum=(50, 70), pet_safe=True,  tip="Heavy feeder - fertilise every 2 weeks in the growing season."),
}
LIGHT_LEVELS = ["low", "medium", "high"]
POT_FACTOR = {"terracotta": 1.25, "plastic": 1.0, "ceramic": 0.9}
SEASON_FACTOR = {"summer": 1.15, "monsoon": 0.85, "winter": 0.8}
SOIL_FEEL = {"dry": 0, "moist": 1, "wet": 2}

# ----------------------------------------------------------------------
# 2. SOIL MOISTURE MODEL
# ----------------------------------------------------------------------
def decay_rate(p, temp, hum, light, pot, season):
    """% soil moisture lost per day."""
    base = (100 - p["threshold"]) / p["water_days"]
    f = (1 + 0.05 * (temp - 24)) * (1 + 0.01 * (50 - hum))
    f *= 1 + 0.10 * (LIGHT_LEVELS.index(light) - 1)
    f *= POT_FACTOR[pot] * SEASON_FACTOR[season]
    return max(base * f, 0.5)

def moisture_after(days, rate):
    return max(100 - rate * days, 0)

# ----------------------------------------------------------------------
# 3. MACHINE LEARNING MODEL  (Decision Tree trained on simulated data)
# ----------------------------------------------------------------------
FEATURES = ["Soil moisture (est.)", "Water threshold", "Soil feel (0-2)",
            "Temperature", "Humidity", "Days since watering"]
LABELS = ["WAIT", "WATER NOW", "OVERWATER RISK"]

def make_label(moist, thr, feel):
    if feel == 2:
        return 2                       # soggy soil: never water, risk of root rot
    if feel == 0 or moist <= thr + 5:
        return 1                       # dry -> water
    return 0

def train_model(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        thr = rng.choice([15, 20, 25, 35, 40, 45, 55])
        days = rng.integers(0, 30)
        temp, hum = rng.uniform(10, 38), rng.uniform(20, 85)
        rate = (100 - thr) / rng.uniform(3, 21) * (1 + 0.05 * (temp - 24)) * (1 + 0.01 * (50 - hum))
        moist = max(100 - rate * days, 0)
        true = make_label(moist, thr, 1)
        # simulated finger-test with noise (people are not perfect sensors!)
        feel = 0 if moist < thr + 5 else (2 if moist > thr + 40 else 1)
        if rng.random() < 0.12:
            feel = rng.integers(0, 3)
        label = make_label(moist, thr, feel)
        X.append([moist, thr, feel, temp, hum, days]); y.append(label)
    X, y = np.array(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    clf = DecisionTreeClassifier(max_depth=6, random_state=1).fit(Xtr, ytr)
    return clf, accuracy_score(yte, clf.predict(Xte))

# ----------------------------------------------------------------------
# 4. SYMPTOM DIAGNOSIS ENGINE  (likelihood table, normalised -> probability)
# ----------------------------------------------------------------------
CAUSES = ["Overwatering", "Underwatering", "Too little light", "Too much direct sun",
          "Low humidity", "Pests (mites/aphids/mealybugs)", "Nutrient deficiency", "Cold stress"]
#                                    Ovr  Und  LowL Sun  Hum  Pest Nut  Cold
SYMPTOMS = {
    "yellow leaves":         [.60, .20, .30, .10, .05, .20, .50, .10],
    "brown crispy tips":     [.05, .45, .05, .35, .60, .05, .15, .10],
    "drooping/wilting":      [.45, .65, .10, .25, .20, .10, .05, .30],
    "leggy/stretched":       [.02, .02, .90, .01, .01, .02, .20, .02],
    "brown/black spots":     [.45, .05, .05, .30, .10, .30, .05, .10],
    "white fluff/webbing":   [.02, .02, .02, .02, .20, .95, .02, .02],
    "mushy stem/bad smell":  [.95, .02, .05, .01, .01, .03, .02, .15],
    "leaf drop":             [.40, .35, .35, .20, .20, .20, .20, .60],
    "pale/faded leaves":     [.10, .10, .40, .60, .05, .10, .55, .10],
    "slow/no growth":        [.20, .20, .50, .10, .10, .10, .70, .40],
}
SYMPTOM_LIST = list(SYMPTOMS)
FIXES = {
    "Overwatering": "Stop watering, let top 3-4 cm dry, check drainage holes; repot in fresh mix if roots are mushy.",
    "Underwatering": "Water slowly until it drains out the bottom; set a reminder using the forecast above.",
    "Too little light": "Move closer to a window (bright, indirect light) or add a grow light for 6-8 h/day.",
    "Too much direct sun": "Shift back from the window or use a sheer curtain, especially for afternoon sun.",
    "Low humidity": "Group plants, use a pebble tray/humidifier, avoid AC or heater vents.",
    "Pests (mites/aphids/mealybugs)": "Isolate the plant, wipe leaves with soapy water or neem oil spray every 5-7 days.",
    "Nutrient deficiency": "Feed balanced liquid fertiliser at half strength every 2-4 weeks in growth season.",
    "Cold stress": "Keep away from cold windows/AC and drafts; ideal above 15-18 C.",
}

def diagnose(selected, soil_feel, moist, thr):
    prior = np.ones(len(CAUSES)) / len(CAUSES)
    # soil evidence nudges the prior (a real expert does this too)
    if soil_feel == "wet" or moist > thr + 40:
        prior[0] *= 2.0
    if soil_feel == "dry" or moist < thr:
        prior[1] *= 2.0
    post = prior.copy()
    for s in selected:
        post = post * np.array(SYMPTOMS[s])
    return post / post.sum()

# ----------------------------------------------------------------------
# 5. HEALTH SCORE
# ----------------------------------------------------------------------
def health_score(p, temp, hum, light, n_symptoms, status):
    score = 100
    lo, hi = p["temp"]
    if temp < lo: score -= min(25, (lo - temp) * 4)
    if temp > hi: score -= min(25, (temp - hi) * 4)
    lo, hi = p["hum"]
    if hum < lo: score -= min(15, (lo - hum) * 0.7)
    if hum > hi: score -= min(10, (hum - hi) * 0.5)
    score -= 15 * abs(LIGHT_LEVELS.index(light) - LIGHT_LEVELS.index(p["light"]))
    score -= 7 * n_symptoms
    if status == "OVERWATER RISK": score -= 12
    if status == "WATER NOW": score -= 6
    return int(max(0, min(100, score)))

# ----------------------------------------------------------------------
# TERMINAL HELPERS
# ----------------------------------------------------------------------
G, Y, R, C, B, X = "\033[92m", "\033[93m", "\033[91m", "\033[96m", "\033[1m", "\033[0m"

def ask(prompt, default=None, cast=str, choices=None):
    while True:
        shown = f" [{default}]" if default is not None else ""
        try:
            raw = input(f"{C}?{X} {prompt}{shown}: ").strip().lower()
        except EOFError:
            raw = ""
        if raw == "" and default is not None:
            raw = str(default)
        try:
            val = cast(raw)
        except ValueError:
            print(f"  {R}Please enter a valid {cast.__name__}.{X}"); continue
        if choices and val not in choices:
            print(f"  {R}Choose one of: {', '.join(map(str, choices))}{X}"); continue
        return val

def bar(value, width=30, color=G):
    n = int(round(width * value / 100))
    return f"{color}{'█' * n}{X}{'░' * (width - n)} {value:.0f}%"

def banner():
    print(f"""{G}{B}
  ╔══════════════════════════════════════════════════════╗
  ║        🌿  AI PLANT CARE ASSISTANT  🌿               ║
  ║   Smart watering • Diagnosis • Health score • Graphs ║
  ╚══════════════════════════════════════════════════════╝{X}""")

# ----------------------------------------------------------------------
# GRAPH DASHBOARD
# ----------------------------------------------------------------------
def make_dashboard(name, p, rate, days_since, thr, temp, hum, light, probs, clf, score, status, fname):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"AI Plant Care Dashboard - {name.title()}   |   Health {score}/100   |   {status}",
                 fontsize=15, fontweight="bold", color="#1b5e20")

    # (a) soil moisture: past + 14 day forecast
    a = ax[0, 0]
    t = np.arange(-days_since, 15)
    m = [100 if d <= -days_since else moisture_after(d + days_since, rate) for d in t]
    a.plot(t, m, color="#2e7d32", lw=2.5, label="Soil moisture")
    fut = np.arange(0, 15)
    a.plot(fut, [max(100 - rate * d, 0) for d in fut], color="#1565c0", ls="--", lw=2,
           label="If you water today")
    a.axhline(thr, color="red", ls="--", label=f"Water threshold ({thr}%)")
    a.axvline(0, color="grey", ls=":", label="Today")
    a.fill_between(t, m, thr, where=np.array(m) > thr, color="#a5d6a7", alpha=.5)
    a.set_title("Soil Moisture Forecast (next 14 days)"); a.set_xlabel("Days (0 = today)")
    a.set_ylabel("Moisture %"); a.set_ylim(0, 105); a.legend(); a.grid(alpha=.3)

    # (b) environment vs ideal range
    a = ax[0, 1]
    labs = ["Temperature (C)", "Humidity (%)"]
    ideals = [p["temp"], p["hum"]]; vals = [temp, hum]
    for i, (rng, v) in enumerate(zip(ideals, vals)):
        a.barh(i, rng[1] - rng[0], left=rng[0], color="#c8e6c9", height=.45,
               label="Ideal range" if i == 0 else None)
        ok = rng[0] <= v <= rng[1]
        a.plot(v, i, "o", ms=14, color="#2e7d32" if ok else "#c62828",
               label="Your value" if i == 0 else None)
        a.text(v, i + .28, f"{v:g}", ha="center", fontweight="bold")
    a.set_yticks([0, 1]); a.set_yticklabels(labs); a.set_xlim(0, 100); a.set_ylim(-.6, 1.7)
    a.set_title(f"Environment vs Ideal (light: yours={light}, ideal={p['light']})")
    a.legend(loc="lower right"); a.grid(alpha=.3, axis="x")

    # (c) diagnosis probabilities
    a = ax[1, 0]
    order = np.argsort(probs)
    cols = ["#c62828" if probs[i] == probs.max() else "#90a4ae" for i in order]
    a.barh([CAUSES[i] for i in order], probs[order] * 100, color=cols)
    a.set_title("Diagnosis: Probable Causes (%)"); a.set_xlabel("Probability %")
    a.grid(alpha=.3, axis="x")

    # (d) ML feature importance
    a = ax[1, 1]
    imp = clf.feature_importances_
    o = np.argsort(imp)
    a.barh([FEATURES[i] for i in o], imp[o] * 100, color="#5c6bc0")
    a.set_title("ML Model: What Drives the Watering Decision"); a.set_xlabel("Importance %")
    a.grid(alpha=.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(fname, dpi=130)
    plt.close(fig)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    banner()
    print(f"\n{B}Training the watering-decision ML model...{X}")
    clf, acc = train_model()
    print(f"  Decision Tree trained on 4000 simulated cases -> test accuracy: {G}{acc*100:.1f}%{X}\n")

    print(f"{B}Plants I know:{X} " + ", ".join(p.title() for p in PLANTS))
    name = ask("Which plant do you have?", "monstera", choices=list(PLANTS))
    p = PLANTS[name]

    print(f"\n{B}Tell me about its environment{X}")
    light = ask("Light it gets (low / medium / high)", "medium", choices=LIGHT_LEVELS)
    temp = ask("Room temperature in C", 30, float)
    hum = ask("Room humidity % (Kerala is humid, ~75)", 75, float)
    pot = ask("Pot type (terracotta / plastic / ceramic)", "plastic", choices=list(POT_FACTOR))
    season = ask("Season (summer / monsoon / winter)", "monsoon", choices=list(SEASON_FACTOR))
    days_since = ask("Days since last watering", 6, int)
    feel = ask("Soil feel when you push a finger 3 cm in (dry / moist / wet)", "moist", choices=list(SOIL_FEEL))

    print(f"\n{B}Any symptoms?{X}  Pick numbers separated by commas, or press Enter for none:")
    for i, s in enumerate(SYMPTOM_LIST, 1):
        print(f"   {i:>2}. {s}")
    raw = input(f"{C}?{X} Symptoms: ").strip()
    chosen = []
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit() and 1 <= int(tok) <= len(SYMPTOM_LIST):
            chosen.append(SYMPTOM_LIST[int(tok) - 1])

    # ---- compute -----------------------------------------------------
    thr = p["threshold"]
    rate = decay_rate(p, temp, hum, light, pot, season)
    moist = moisture_after(days_since, rate)
    days_left = (moist - thr) / rate
    x = np.array([[moist, thr, SOIL_FEEL[feel], temp, hum, days_since]])
    pred = clf.predict(x)[0]; conf = clf.predict_proba(x)[0].max() * 100
    status = LABELS[pred]
    probs = diagnose(chosen, feel, moist, thr)
    score = health_score(p, temp, hum, light, len(chosen), status)

    # ---- report ------------------------------------------------------
    line = "═" * 62
    print(f"\n{G}{B}{line}\n  📋  CARE REPORT: {name.upper()}\n{line}{X}")
    print(f"  Environment : {temp:g}C, {hum:g}% humidity, {light} light, {pot} pot, {season}")
    print(f"  Water use   : loses ~{rate:.1f}% soil moisture per day in your conditions")
    print(f"  Soil moisture now (estimated): {bar(moist, color=G if moist > thr else R)}")
    print(f"  Watering threshold           : {thr}%")

    col = {"WAIT": G, "WATER NOW": Y, "OVERWATER RISK": R}[status]
    print(f"\n  {B}🤖 ML Decision : {col}{status}{X}  (confidence {conf:.0f}%)")
    if status == "WATER NOW":
        print("     -> Water thoroughly until water drains from the bottom.")
    elif status == "OVERWATER RISK":
        print("     -> Soil is too wet. Do NOT water. Check drainage and airflow.")
    else:
        print(f"     -> Next watering in about {max(days_left, 0):.1f} day(s).")

    print(f"\n  {B}💚 Health Score:{X} {bar(score, color=G if score >= 70 else (Y if score >= 45 else R))}")
    issues = []
    if not p["temp"][0] <= temp <= p["temp"][1]: issues.append(f"Temperature outside ideal {p['temp'][0]}-{p['temp'][1]} C")
    if not p["hum"][0] <= hum <= p["hum"][1]: issues.append(f"Humidity outside ideal {p['hum'][0]}-{p['hum'][1]} %")
    if light != p["light"]: issues.append(f"Light is '{light}', plant prefers '{p['light']}'")
    for i in issues: print(f"     {Y}! {i}{X}")
    if not issues: print(f"     {G}✔ Environment matches this plant's needs{X}")

    if chosen:
        top = np.argsort(probs)[::-1][:3]
        print(f"\n  {B}🩺 Diagnosis{X} (symptoms: {', '.join(chosen)})")
        for r, i in enumerate(top, 1):
            print(f"     {r}. {CAUSES[i]:<34} {probs[i]*100:5.1f}%")
        print(f"     {B}Fix:{X} {FIXES[CAUSES[top[0]]]}")
    else:
        print(f"\n  {B}🩺 Diagnosis:{X} no symptoms reported - plant looks fine!")

    print(f"\n  {B}💡 Pro tip:{X} {p['tip']}")
    print(f"  {B}🐾 Pet safety:{X} {'Non-toxic to pets' if p['pet_safe'] else 'Toxic to cats/dogs if chewed - keep out of reach'}")

    fname = "plant_dashboard.png"
    make_dashboard(name, p, rate, days_since, thr, temp, hum, light, probs, clf, score, status, fname)
    print(f"\n  {G}📊 Graph dashboard saved -> {fname}{X}")
    print(f"{G}{B}{line}{X}\n  Thank you for using AI Plant Care Assistant! 🌱\n")

if __name__ == "__main__":
    main()
