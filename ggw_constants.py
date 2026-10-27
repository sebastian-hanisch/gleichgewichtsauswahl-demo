"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): Geburts-Sterbe-Kette O(n) im Logarithmus (n = 1 000 in Millisekunden); Schwelle x*(p*, n) 40 Bisektionsschritte à O(n);
Ratenfunktion O(Gitter) mit 200 000 Zellmitten; Kette über drei Formate O(Zustände³) mit 231 Zuständen bei n = 20 (E5 rechnet n = 12 mit 91 Zuständen, 160 Spiele, auf Knopfdruck);
Einzelspieler-Simulation O(Schritte) in Python (3·10^6 Schritte rund 1 bis 2 Sekunden, auf Knopfdruck)."""

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
PAY_MIN, PAY_MAX = -10, 10                  # Zeitersparnis je Sendung in Minuten; negativ = Zeit verloren
N_MIN, N_MAX, DEFAULT_N = 4, 1000, 50       # Betriebe in der Gruppe; O(n) je Kette
X_MIN, X_MAX, DEFAULT_X = 0.25, 40.0, 6.0   # Schärfe x = beta·lam (Einheiten der Auszahlungsskala); beta = x/lam je Minute
MODELS = ("Logit", "Gleichmäßige Fehler", "Typabhängige Fehler")
EPS_EXP_MIN, EPS_EXP_MAX = 0.5, 8.0         # eps = 10^(-e): von 0,32 bis 10^-8
DEFAULT_EPS_EXP, DEFAULT_EPS_S_EXP, DEFAULT_EPS_H_EXP = 1.0, 1.0, 3.0
START_MIN, START_MAX, DEFAULT_START = 0, 100, 50       # Anfangsanteil der S-Betriebe der Simulation in Prozent
SIM_STEPS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)      # Budget der Einzelspieler-Simulation (Aktualisierungen)
DEFAULT_SIM_STEPS = 100_000
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 10, 200, 60          # Läufe der Austrittszeit-Simulation (E6)
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- feste Annahmen -------------------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100001, 100002, 100003, 100004)      # feste Zufallsströme der Messreihen: nie ein einzelner Seed als Befund
THRESHOLD_LEVEL = 0.99                              # Masse „fast alle S“, ab der die Konvention „steht“
THRESHOLD_PSTARS = ("1/10", "1/4", "2/5", "9/20")   # p* der Schwellentabelle (Brüche)
THRESHOLD_NS = (10, 20, 50, 100, 200, 1000)
SURFACE_XS = (1, 2, 4, 6, 8, 12)                    # Schärfen der Massentabelle in E1
INVARIANCE_GAMES = ((5, 2, 1, 2), (4, 1, 0, 1), (3, 5, 4, 0), (13, 9, 8, 10), (9, 100, 99, 6))    # alle mit a - d = 3 und b - c = 1
PHASE_N = 2000                                      # Gruppengröße des Vergleichs Modus gegen Ratenfunktion
PHASE_NS = (50, 200, 1000, 5000)                    # Gruppengrößen der Mitte-Masse bei p* = 1/2
THREE_N = 12                                        # Betriebe im Drei-Formate-Experiment (91 Zustände)
THREE_GAMES_PER_STREAM = 40
THREE_X = 20.0                                      # beta·(größte Diagonale von Sym) im Drei-Formate-Experiment
THREE_EPS = 1e-4
DISSENTER_EXAMPLES = ((100003, 106), (100003, 140))      # (Strom, Index im Strom, ab 0): Spiele, in denen Logit und gleichmäßige Fehler verschieden wählen (gefunden in einer Suche, im Test erneut geprüft)
SIM_BURN_FRACTION = 0.2                             # Einschwingschritte der Simulation: ein Fünftel des Budgets
HITTING_CAP = 2_000_000                             # Obergrenze der Einzelaktualisierungen je Lauf der Austrittszeit-Simulation

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_S = "#2F6B65"          # neues Format S
COLOR_H = "#C0762B"          # altes Format H
COLOR_MID = "rgba(150,150,150,0.65)"
COLOR_LINE = "#4c78a8"
COLOR_RED = "#d62728"

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {
    "a": 5, "b": 2, "c": 1, "d": 2,
    "n": DEFAULT_N, "x": DEFAULT_X, "model": MODELS[0], "eps_exp": DEFAULT_EPS_EXP, "eps_s_exp": DEFAULT_EPS_S_EXP, "eps_h_exp": DEFAULT_EPS_H_EXP,
    "start": DEFAULT_START, "sim_steps": DEFAULT_SIM_STEPS, "runs": DEFAULT_RUNS, "seed": DEFAULT_SEED,
}
PRESETS = {
    "Neues Palettenformat (5, 2, 1, 2)": dict(BASE),
    "Bündeln am Hub (6, 4, 0, 4)": {**BASE, "a": 6, "b": 4, "c": 0, "d": 4},
    "Gleichauf (6, 4, 2, 4)": {**BASE, "a": 6, "b": 4, "c": 2, "d": 4},
    "Gleiches Spiel, anderes Gefälle (3, 5, 4, 0)": {**BASE, "a": 3, "b": 5, "c": 4, "d": 0},
    "Zu wenig Schärfe (x = 2)": {**BASE, "x": 2.0},
}
PRESET_HELP = {
    "Neues Palettenformat (5, 2, 1, 2)": "Das neue Format ist auszahlungs- und risikodominant: p* = 1/4. Bei x = 6 und n = 50 steht die Konvention „neues Format“ (Experiment E1).",
    "Bündeln am Hub (6, 4, 0, 4)": "Das neue Format zahlt beim Zusammentreffen mehr (6 gegen 4), aber p* = 2/3: Das alte Format ist risikodominant. Bei x = 6 und n = 50 steht die Konvention „altes Format“.",
    "Gleichauf (6, 4, 2, 4)": "Beide Formate gleich riskant (p* = 1/2): Das neue zahlt beim Zusammentreffen mehr, aber die Lücken sind gleich. Hier gibt es einen kritischen Wert x = 4 (Experiment E3); bei x = 6 und n = 50 liegt die Masse auf beiden Seiten.",
    "Gleiches Spiel, anderes Gefälle (3, 5, 4, 0)": "Dieselben Lücken wie im ersten Preset (a − d = 3, b − c = 1), aber das alte Format zahlt beim Zusammentreffen mehr (5 gegen 3). Die Verteilung ist dieselbe (Experiment E2).",
    "Zu wenig Schärfe (x = 2)": "Das erste Spiel mit Schärfe x = 2 statt 6: Bei n = 50 liegt die Masse fast ganz in gemischten Zuständen, es entsteht keine Konvention (Experiment E1).",
}
