# Welche Konvention überlebt? Auswahl durch Rauschen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-gleichgewichtsauswahl-demo.streamlit.app/)** -->

Zweites Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel: **Palettenformat.** Viele Betriebe wählen das neue Format (S) oder das alte (H); die Auszahlung ist die Zeitersparnis je Sendung in Minuten, und es lohnt sich nur, dasselbe zu wählen wie die anderen. Jeder Betrieb entscheidet sich immer wieder neu und liegt gelegentlich daneben (Logit: die bessere Wahl ist wahrscheinlicher, nie sicher). Die Demo rechnet **exakt**, auf welche Konvention die Gruppe auf Dauer läuft, **wie scharf die Wahl sein muss**, damit überhaupt eine entsteht, was sich für **große Gruppen** ändert, ob das Ergebnis am **Fehlermodell** hängt und wann eine **Simulation** die Antwort nicht mehr erreicht.

Abgrenzung: `koordinationsspiel-demo` (erstes Stück) zeigt Gleichgewichte, Einzugsbereiche und die Identität „Logit-Kette = Gibbs-Maß“ für eine feste Gruppe; hier ist diese Identität nur das Werkzeug (Orakel), und die Frage ist die Auswahl über Schärfe, Gruppengröße und Fehlermodell. `replikator-demo` rechnet Moran-Prozesse für Gefangenendilemma-Automaten, `rauschen-kooperation-demo` die Fehlerrate im wiederholten Spiel mit zwei Partnern; keine von beiden wählt zwischen Koordinationsgleichgewichten. **Eigennutz und die Lernregel sind Modellannahmen**, keine Aussagen über Menschen oder Terminals; die Auszahlungen sind gewählt, nicht gemessen.

## Kernfrage

*Wenn sich viele Betriebe immer wieder neu entscheiden und gelegentlich daneben liegen: auf welche Konvention läuft das System auf Dauer, und wie scharf muss die Wahl sein, damit überhaupt eine Konvention entsteht?* Risikodominanz, Potentialspiele und die Auswahl durch Rauschen sind Lehrbuchstoff (Quellen unten). Eigenes dieser Demo sind (1) die **Auswahl als Fläche** über Schwelle p\*, Gruppengröße und Schärfe, (2) die **Invarianz**: im Logit zählt nur die Skala λ und die Schwelle p\*, die Auszahlungsdominanz hat keinen Einfluss, (3) die **Phasenstruktur** großer Gruppen (Ratenfunktion, kritischer Wert 4 bei p\* = 1/2), (4) der **Vergleich der Fehlermodelle** mit exakter Zählformel und dem Kippen bei typabhängigen Fehlern und (5) die **Rechengrenze**: die mittlere Zeit, ein Becken zu verlassen, in Zehnerpotenzen, gegen Einzelspieler-Simulation.

## Befunde und Korrekturen gegenüber dem Plan

Die Hypothesen des Einzelplans (`koordinationsspiele-planung/STUECK-2-gleichgewichtsauswahl.md`) wurden im Bau neu gerechnet; die Vormessung des Plans (Wegwerfskripte mit numpy-Zufall) liefert keine README-Zahl. Die Schärfe ist **x = β·λ** mit λ = (a − d) + (b − c); „fast alle S“ heißt: mindestens 90 % der Betriebe nutzen S, „fast alle H“: höchstens 10 %. Eine Konvention „steht“, wenn mindestens 99 % der Masse bei einer der beiden Seiten liegt.

| | Hypothese | Ergebnis | Zahl (diese Demo; Test) |
|---|---|---|---|
| H1 | Stationäre Verteilung der Geburts-Sterbe-Kette = Gibbs-Maß über alle Konfigurationen mal Binomialfaktor = Eigenvektor der vollen Matrix | **Ja** | 300 Zufallsspiele (n = 4 bis 40, x = 0,2 bis 12): Abweichung gegen Grassmann–Taksar–Heyman an der vollen Übergangsmatrix und gegen die Zählformel C(n, k)·exp(βΦ) jeweils unter 10⁻¹³; gegen den numpy-Eigenvektor unter 10⁻⁵ (er löst kleine Wahrscheinlichkeiten schlecht auf, nur Näherung). 60 Zufallsspiele mit n = 4 bis 10 gegen die Aufzählung **aller 2ⁿ Konfigurationen** mit dem Paarpotential: unter 10⁻¹³. Binomialfaktor: bei (5, 2, 1, 2), n = 10, x = 2 hat Zustand k = 5 die Masse 0,0978 auf 252 Konfigurationen, k = 10 nur 0,0760 auf einer; je Konfiguration liegt k = 10 um den Faktor 196 vorn (`test_h1_*`) |
| H2 | Im Logit zählt nur (λ, p\*); die Auszahlungsdominanz a − b hat keinen Einfluss | **Ja** | Fünf Spiele mit a − d = 3, b − c = 1 (λ = 4, p\* = 1/4) und a − b = 3, 3, −2, 4, −91: größte Abweichung der Verteilungen (n = 30, β = 1) unter 10⁻¹⁴ (Rundung der Gleitkommazahlen); Masse „fast alle S“ bei allen 0,869 (`test_h2_*`) |
| H3 | Das risikodominante Format gewinnt ab x ≥ 8 (p\* ≤ 0,4, n ≥ 10); die Plan-Fassung „β ≥ 2“ gilt nur bei λ = 4 | **Ja, mit Korrektur** | x = 8: Masse „fast alle S“ mindestens 0,99 in **18 von 18** Zellen (p\* ∈ {1/10, 1/4, 2/5} × n ∈ {10, 20, 50, 100, 200, 1 000}); die schlechteste Zelle ist p\* = 2/5, n = 10 mit 0,9915. β = 2 ist nur bei λ = 4 gleich x = 8: im Spiel (2, 1, 0, 0) (λ = 3, p\* = 1/3) ist β = 2 nur x = 6, die Masse bei n = 10 beträgt 0,970, die Schwelle 6,92 (`test_h3_*`) |
| H4 | Unterhalb der Schwelle entsteht keine Konvention, und größere Gruppen machen es schlimmer | **Ja** | p\* = 1/4, x = 2: Masse „fast alle S“ 0,246 (n = 10), unter 0,001 (n = 100), unter 10⁻⁹ (n = 1 000); die Mitte hat 0,751, mehr als 0,999 und mehr als 1 − 10⁻⁹. Vollständige Tabelle unten (`test_h4_*`) |
| H5 | Die Schwelle x\* sinkt mit n und steigt mit p\* | **Ja** | Tabelle x\* unten: 18 von 18 benachbarten Paaren steigen mit p\*, 20 von 20 fallen mit n; die Masse wächst in x auf dem ganzen Raster 0,25 bis 40 (Schritt 0,25) für p\* ∈ {1/10, 1/4, 2/5, 9/20}, n ∈ {10, 100}, damit ist die Bisektion zulässig. Der Umschlag in p\* ist schroff (n = 100, x = 8): Masse „fast alle S“ 0,999 bei p\* = 0,49, 0,500 bei 0,50, 0,000 bei 0,51; p\* = 1/2 teilt auf 10⁻¹² genau (`test_h5_*`) |
| H6 | Bei p\* = 1/2 gibt es nur oberhalb des kritischen Werts x = 4 zwei Konventionen, darunter nur die Mitte | **Ja** | Zahl der Logit-Gleichgewichte (Lösungen von y = σ(x·(y − p\*))): 1 für x ≤ 3,9, **3** für x ≥ 4,1; kritischer Wert durch Bisektion 4,000 (±0,001). Masse in der Mitte (0,4·n bis 0,6·n) für x = 3 / 4 / 4,5: n = 50: 0,573 / 0,311 / 0,156; n = 200: 0,859 / 0,434 / 0,066; n = 1 000: 0,999 / 0,644 / 0,000; n = 5 000: 1,000 / 0,885 / 0,000 (`test_h6_*`) |
| H7 | Der Modus der Verteilung ist das Maximum der Ratenfunktion g, also ein Logit-Gleichgewicht | **Ja** | n = 2 000, p\* ∈ {0,25; 0,4}, x = 1 bis 20 (18 Zellen): größter Abstand zwischen Modus und Maximum von g unter 0,0006 (z. B. p\* = 0,4, x = 6: 0,9685 gegen 0,9679). Das zweite lokale Maximum (der unterlegene Ast) entsteht bei p\* = 0,25 ab x = 14,36, bei p\* = 0,4 ab x = 6,89 (`test_h7_*`) |
| H8 | Gleichmäßige Fehler wählen in 2×2 dieselbe Seite wie der Logit; die Steigung ist die Zählung (Zahl Δ > 0) − (Zahl Δ < 0) | **Ja, mit einer Ausnahme (Gleichstand der Zählung)** | Steigung von log₁₀(π(alle S)/π(alle H)) je Dekade von 1/ε: 10,00 / 24,00 / 4,00 / 10,00 / 4,00 gegen die Zählung 10 / 24 / 4 / 10 / 4 in 5 von 5 Zellen. Gleiche Seite in **60 von 63** Zellen (7 Werte von p\* × n ∈ {10, 20, 50} × ε ∈ {0,1; 0,01; 0,001}, Logit bei x = 30); die drei Ausnahmen sind eine Zelle (p\* = 9/20, n = 10) mit Gleichstand der Zählung (5 gegen 5): gleichmäßige Fehler geben beiden Seiten je 0,457 bis 0,500, der Logit wählt S. In der endlichen Gruppe (n = 20) unterscheiden gleichmäßige Fehler p\* = 1/4 und 2/5 kaum (Masse „fast alle S“ 0,405 / 0,677 / 0,925 bei ε = 0,3 / 0,2 / 0,1, beide Werte höchstens 0,0005 auseinander), der Logit bei x = 4 stark (0,857 gegen 0,487) (`test_h8_*`) |
| H9 | Typabhängige Fehlerraten kippen die Auswahl bei der gezählten Schwelle | **Ja, die Schwelle ist exakt 1/730 statt 10⁻³** | p\* = 1/4, n = 40, ε_S = 0,1: 30 Zustände mit Δ > 0, 10 mit Δ < 0; die Auswahl kippt bei ε_H = 1/(1 + 9³) = 1/730 ≈ 1,4·10⁻³ (Zählformel N₊·ln((1 − ε_S)/ε_S) = N₋·ln((1 − ε_H)/ε_H)). Masse bei S 0,629 für ε_H ≥ 0,003, bei ε_H = 10⁻³ 0,463 / 0,264 / 0,273 (S / H / Mitte), Masse bei H mehr als 0,9999 für ε_H ≤ 3·10⁻⁴ (`test_h9_*`) |
| H10 | Drei Formate: Logit = Potentialgipfel, Logit = gleichmäßige Fehler, beide ≠ Auszahlungsdominanz | **Ja für das Potential, selten Abweichler bei den Fehlerregeln** | 160 Spiele (4 Ströme × 40, n = 12, ganzzahlige symmetrische Matrizen 0 bis 9 mit strengen Gleichgewichten, Logit-Schärfe 20, ε = 10⁻⁴): Logit wählt die größte Diagonale in **160 von 160**, Logit und gleichmäßige Fehler dasselbe in 160 von 160, **kein Abweichler**. Die Auszahlungsdominanz stimmt mit dem Logit in 74 von 145 Fällen überein (in 15 Spielen gibt es einen Gleichstand an der Spitze), weicht also in 71 ab. Eine Suche über 4 × 600 = 2 400 Spiele findet **12** mit verschiedener Wahl (und 2, in denen eine Regel nicht eindeutig wählt): ein Existenznachweis, kein Häufigkeitsurteil für andere Spielklassen oder größere n (`test_h10_*`) |
| H11 | Die Simulation erreicht die stationäre Verteilung bei großen Zeiten nicht | **Ja** | Mittlere Zahl der Aktualisierungen je Betrieb bis zur Barriere k\* (erster Zustand, in dem S die beste Antwort ist), p\* = 1/4, im Logarithmus, aus „alle S“ für x = 2 / 4 / 8: n = 10: 10^1,31 / 10^2,85 / 10^7,52; n = 20: 10^2,36 / 10^5,55 / 10^14,98; n = 100: 10^10,68 / 10^26,60 / 10^73,36; aus „alle H“ nur 10^−0,12 bis 10^0,35. Formel gegen Einzelspieler-Simulation (4 Ströme × 100 Läufe, acht Zellen): alle acht innerhalb von 3 Standardfehlern, die größte Abweichung über 2. Bei n = 20, x = 4 braucht der Austritt aus „alle S“ 10^5,55 Aktualisierungen je Betrieb, das größte Budget (3·10⁶ Aktualisierungen) sind 150 000 je Betrieb (`test_h11_*`) |
| H12 | Andere Dynamik (gleichzeitige Aktualisierung), zwei Rollen, heterogene Betriebe, Netze | **Nicht gemessen** | Die Demo kennt nur die asynchrone Aktualisierung (ein Betrieb je Schritt) einer vollständig durchmischten Gruppe. Netze und die Wechselzeit der Konvention sind nicht Gegenstand dieser Demo |

**Masse „fast alle S“ bei p\* = 1/4** (Zeilen n, Spalten x; in Klammern die Mitte):

| n | x = 1 | x = 2 | x = 4 | x = 6 | x = 8 | x = 12 |
|---|---|---|---|---|---|---|
| 10 | 0,060 (0,932) | 0,246 (0,751) | 0,852 (0,148) | 0,989 (0,011) | 0,999 (0,001) | 1,000 (0,000) |
| 100 | 0,000 (1,000) | 0,000 (1,000) | 0,941 (0,059) | 1,000 (0,000) | 1,000 (0,000) | 1,000 (0,000) |
| 1 000 | 0,000 (1,000) | 0,000 (1,000) | 1,000 (0,000) | 1,000 (0,000) | 1,000 (0,000) | 1,000 (0,000) |

**Schwelle x\*** (kleinste Schärfe mit Masse „fast alle S“ ≥ 0,99; Bisektion, `data/threshold_table.json`):

| p\* | n = 10 | 20 | 50 | 100 | 200 | 1 000 |
|---|---|---|---|---|---|---|
| 1/10 | 4,97 | 4,50 | 3,93 | 3,59 | 3,35 | 3,01 |
| 1/4 | 6,06 | 5,49 | 4,79 | 4,39 | 4,09 | 3,70 |
| 2/5 | 7,85 | 7,03 | 6,14 | 5,64 | 5,27 | 4,78 |
| 9/20 | 9,96 | 7,81 | 6,78 | 6,23 | 5,83 | 5,30 |

In den gerechneten Zellen gilt: Je knapper die Risikodominanz, desto schärfer muss gewählt werden; je größer die Gruppe, desto weniger. Für p\* ≥ 1/2 erreicht die Masse bei „fast alle S“ die 99 % bis x = 400 nicht (geprüft bei p\* = 1/2, 2/3 und 3/4).

**Das auszahlungsdominante Format verliert** (Spiel (1, p\*, 0, p\*): H zahlt sicher p\*, S zahlt 1 gegen S und 0 gegen H, also das 1/p\*-fache; n = 50): Masse bei H bei x = 4 für p\* = 0,55 / 0,6 / 0,75 / 0,9 (S zahlt das 1,82- / 1,67- / 1,33- / 1,11-fache): 0,138 / 0,367 / 0,895 / 0,993; bei x = 12 und x = 40 mehr als 0,99999999 bzw. 1,000; die Masse bei S ist in allen zwölf Zellen unter 0,00004 (`test_h5_payoff_dominant_*`). Nur für diese Spielfamilie gerechnet.

**Standardeinstellung** ((5, 2, 1, 2), n = 50, x = 6, Logit): Austritt aus „alle S“ nach 10^24,48 Aktualisierungen je Betrieb, aus „alle H“ nach 1,31; Masse im S-Becken mehr als 99,9 %. Die Simulation mit 100 000 bzw. 3 000 000 Aktualisierungen (Start bei 25 S-Betrieben) hat Gesamtvariationsabstand 0,020 bzw. 0,007 zur exakten Verteilung, **bleibt aber im S-Becken**: ein kleiner Abstand sagt hier nur, dass die Masse im erreichten Becken liegt (`test_h11_default_*`).

### Korrekturen gegenüber dem Plan

- **Invarianz:** Der Plan nennt 1,6·10⁻²² als größte Abweichung; gemessen sind unter 10⁻¹⁴ (Gleitkommarundung der Raten, die Verteilungen sind mathematisch gleich). Der Plan-Wert ist nicht reproduziert; die Ursache der Differenz ist nicht untersucht.
- **„Identisch“ bei gleichmäßigen Fehlern:** Die Vormessung schrieb, die Masse bei S sei für p\* = 1/4 und 2/5 „identisch“; gemessen unterscheidet sie sich um bis zu 0,0005 (auf drei Stellen gleich, nicht identisch).
- **Kippschwelle typabhängiger Fehler:** Der Plan sagt „vorhergesagt 10⁻³“ aus der Näherung mit ln(1/ε); die exakte Zählformel mit den Wahlverhältnissen (1 − ε)/ε gibt 1/730 ≈ 1,4·10⁻³. Beide liegen zwischen den gemessenen Seiten (0,003 und 3·10⁻⁴).
- **Gleiche Seite bei gleichmäßigen Fehlern und Logit** gilt nicht, wenn die Zählung Δ > 0 gegen Δ < 0 ein Gleichstand ist (H8).
- **Zweiter Ast bei p\* = 0,4:** Der Plan nannte x = 8; das zweite lokale Maximum der Ratenfunktion entsteht schon bei x = 6,89 (bei p\* = 0,25 bei 14,36).
- **Drei Formate:** Die Vormessung fand 2 Abweichler in 160 Spielen (318 von 320 gleich über zwei Spielklassen) und 51 von 160 Abweichungen der Auszahlungsdominanz; mit den eigenen SplitMix64-Strömen und einer Spielklasse (symmetrisch, Spaltenverschiebung) gibt es 0 Abweichler in 160 Spielen und 71 von 145 Abweichungen der Auszahlungsdominanz. Die Abweichler stehen erst in der längeren Suche (12 in 2 400).
- **Formel gegen Simulation:** Der Plan sagte „innerhalb etwa eines Standardfehlers“; mit eigenen Strömen liegen alle acht Zellen innerhalb von 3, nicht von 1 Standardfehlern. Ein größerer Kontrolllauf außerhalb des Repos (2 000 Läufe der Zelle n = 10, x = 4, aus „alle S“) lag innerhalb eines Standardfehlers der Formel; er steht nicht in den Tests.
- **Austritt aus „alle H“:** etwa eine Aktualisierung je Betrieb (10^−0,12 bis 10^0,35), das H-Becken ist klein.
- **Port:** Der Plan schlug 9204 vor; das war beim Bau frei (TCP-Prüfung) und steht in `.claude/launch.json`.
- **Aufzählung aller 2ⁿ Konfigurationen** läuft im Repo bis n = 10 (im Plan bis 14), damit der Test unter 10 Sekunden bleibt.
- **Quellen** aus dem Plan sind nur dort aufgenommen, wo ein Abstract geprüft wurde (siehe unten).

## Modell

Symmetrisches 2×2-Spiel (a, b, c, d): a beide S, b beide H, c ich S und der andere H, d ich H und der andere S. n Betriebe, vollständig durchmischt; Zustand ist die Zahl k der S-Betriebe. Ein gleichverteilt gezogener Betrieb entscheidet neu; sein Vorteil von S gegenüber H bei j anderen S-Betrieben ist Δⱼ = ((a − d)·j − (b − c)·(n − 1 − j))/(n − 1) = λ·(j/(n − 1) − p\*). **Logit:** Wahl von S mit 1/(1 + e^(−βΔ)); **gleichmäßige Fehler:** beste Antwort, mit Wahrscheinlichkeit ε gleichverteilt zwischen S und H; **typabhängig:** ist S beste Antwort, wird mit ε_S trotzdem H gewählt, ist H beste Antwort, mit ε_H trotzdem S; bei Δ = 0 je 1/2. Die Kette über k ist eine Geburts-Sterbe-Kette, ihre stationäre Verteilung folgt aus Detailed Balance, im Logarithmus gerechnet. Beim Logit ist sie π(k) ∝ C(n, k)·exp(βΦ(k)) mit dem Paarpotential P(S, S) = a − d, P(H, H) = b − c, P(S, H) = 0 und Φ(k) = [(a − d)·k(k − 1)/2 + (b − c)·(n − k)(n − k − 1)/2]/(n − 1). **Große Gruppen:** mit x = β·λ und y = k/n ist log π(k) ≈ n·g(y), g(y) = x·(y²/2 − p\*·y) + H(y). **Zeit:** mittlere Zahl der Aktualisierungen von Zustand 0 bis zur Barriere k\* aus der Formel der Geburts-Sterbe-Kette, durch n geteilt (je Betrieb). Der Regler **x** stellt die Schärfe in Einheiten der Auszahlungsskala ein (β = x/λ); im Nicht-Koordinationsspiel ist die Skala |a − d| + |b − c|. Formeln: 📐-Expander der App.

## Methodik

| Baustein | Datei |
|---|---|
| Klassen, Schwelle p\*, Vorteil Δⱼ, exakte Vorzeichenzählung, Barriere, Potential | `ggw_game.py` |
| Raten (Logit, gleichmäßig, typabhängig), stationäre Verteilung, geschlossene Form, Massen, Schwelle x\* | `ggw_chain.py` |
| Ratenfunktion, Logit-Gleichgewichte, kritischer Wert, zweiter Ast | `ggw_limit.py` |
| Kette über Zusammensetzungen (drei Formate), Grassmann–Taksar–Heyman | `ggw_three.py` |
| Einzelspieler-Simulation, mittlere Austrittszeit im Logarithmus | `ggw_sim.py` |
| Zufallsspiele mit drei Formaten, Startzustand | `ggw_scenario.py`, `ggw_rng.py` (SplitMix64, wörtliche Kopie aus `ab-test-demo/abt_rng.py`) |
| Die Messreihen der sechs Experimente E1 bis E6, vorgerechnete Schwellentabelle | `ggw_experiments.py`, `data/threshold_table.json`, `tools/make_threshold_table.py` |

Alle Zufallszahlen kommen aus SplitMix64 (gleiche Eingabe, gleiche Zahlen auf jeder Plattform). Die Verteilungen sind **exakt gerechnet**, nicht simuliert; Simulation gibt es nur in E6 und im Orakel. Die Schwellentabelle liegt vorgerechnet in `data/threshold_table.json` mit einem Hash der Quellen (`ggw_game.py`, `ggw_chain.py`); ist er veraltet, rechnet die App die Tabelle neu und `tests/test_threshold_table.py` schlägt an. Laufzeit: Die Ketten bis n = 1 000 rechnen beim Laden der Seite; das Drei-Formate-Experiment (160 Ketten) und beide Simulationen in E6 rechnen erst auf Knopfdruck.

## Ehrliche Grenzen

- **Alle Aussagen gelten für diese Lernregeln** (Logit, gleichmäßige und typabhängige Fehler) in einer vollständig durchmischten Gruppe mit asynchroner Aktualisierung, nicht für Menschen oder Terminals. Auswahl und Fehlermodell hängen zusammen (E4): mit typabhängigen Fehlern kippt die Auswahl.
- Die Aussage „die Auszahlungsdominanz zählt nicht“ gilt für den Logit mit gleicher Schärfe für alle; bei der Gegenprobe „das auszahlungsdominante Format verliert“ ist nur die Spielfamilie (1, p\*, 0, p\*) gerechnet.
- Die Drei-Formate-Zahlen gelten für **n = 12**, **eine** Spielklasse (symmetrische ganzzahlige Matrizen 0 bis 9 mit strengen Gleichgewichten und Spaltenverschiebung) und einen Logit-Schärfewert; die Abweichler sind ein Existenznachweis.
- Die Schwelle x\* gilt für das Niveau 99 % der Masse bei „fast alle S“ (mindestens 90 % der Betriebe); mit anderem Niveau oder anderer Grenze verschieben sich die Zahlen.
- Das Maximum von g beschreibt den Modus für n → ∞; bei n = 2 000 liegt der Abstand unter 0,0006 (H7), für andere n nicht gemessen.
- Austrittszeiten sind Erwartungswerte; die Simulation sieht sie nur bis zur Obergrenze von 3·10⁶ Aktualisierungen.
- Ein kleiner Gesamtvariationsabstand der Simulation beweist nichts über das andere Becken, wenn das Budget unter der Austrittszeit liegt (Standardeinstellung).
- Mausklicks auf die Diagramme sind nicht getestet (die Diagramme haben keine Auswahl).
- Nur an erfundenen Auszahlungen geprüft; keine Messung an Terminals oder Menschen.

## Quellen

Am Abstract oder an der bibliographischen Angabe geprüft (nicht im Volltext gelesen): Kandori, Mailath, Rob (1993), „Learning, Mutation, and Long Run Equilibria in Games“, *Econometrica* 61(1), 29–56 (Abstract: in 2×2-Spielen mit zwei strikten symmetrischen Gleichgewichten wird bei großer Population das risikodominante gewählt). Young (1993), „The Evolution of Conventions“, *Econometrica* 61(1), 57–84 (Abstract: stochastisch stabile Gleichgewichte; in 2×2-Spielen stimmen sie mit den risikodominanten überein, im allgemeinen Fall nicht). Blume (1993), „The Statistical Mechanics of Strategic Interaction“, *Games and Economic Behavior* 5(3), 387–424 (Abstract: strategische Interaktion auf einem Gitter, stochastische Wahlregeln, stationäre Verteilungen; nur die bibliographische Angabe und diese Beschreibung geprüft). Monderer, Shapley (1996), „Potential Games“, *Games and Economic Behavior* 14, 124–143 (Abstract: Potentialfunktionen für Spiele in Normalform und deren Charakterisierung). **Nicht geprüft und nicht verwendet:** Bergin, Lipman (1996), Marden, Shamma (log-lineares Lernen), Harsanyi, Selten (1988). Die Formel π(k) ∝ C(n, k)·exp(βΦ(k)), die Invarianz in (λ, p\*), die Ratenfunktion und der kritische Wert 4 sind eigene Rechnung, durch Aufzählung, volle Übergangsmatrix und Gitter bestätigt, nicht aus Quellen.

## Verifikation

- **Kern gegen Handrechnung** (`tests/conftest.py`, `test_game.py`, `test_chain.py`, `test_limit.py`, `test_three.py`, `test_sim.py`): n = 2 (π ∝ (1, 2e^(−β), e^(2β))), n = 3 (π ∝ (1, 3e^(−β), 3, e^(3β))), Δ und Potential von Hand, Invarianz, Spiegelung p\* ↔ 1 − p\*, Zählung für n = 5 (3 positive, 1 negative, 1 Gleichstand), Kettenzeit bei β = 0 (6 Aktualisierungen), Austrittszeit für zwei Betriebe (1 + e), GTH an einer Kette mit Übergang 10⁻⁶⁰.
- **Orakel** (`tests/test_oracle_gleichgewichtsauswahl.py`): stationäre Verteilung gegen GTH an der vollen Matrix (eigene Fassung im Test), numpy-Eigenvektor, geschlossene Form und exakte Bruchrechnung der Fehlermodelle; Aufzählung aller 2ⁿ Konfigurationen und der Nachweis, dass das Paarpotential ein Potential der Auszahlungen ist (Brüche); Einzelspieler-Simulation mit Auszahlung aus dem ganzen Zustandsvektor (numpy-Strom); Modus gegen Maximum von g; Zahl der Logit-Gleichgewichte gegen Iteration der Antwortabbildung; Austrittszeit gegen das lineare Gleichungssystem der vollen Kette und gegen Simulation; Kette über Zusammensetzungen gegen die Geburts-Sterbe-Kette und gegen exakte Brüche bei drei Formaten. Die große Fassung (300 Zufallsspiele, n bis 40) lief im Scratch-Ordner und steht hier in `test_claims.py`.
- **Randfälle** (`test_edge_cases.py`): n = 2, kein Betrieb allein, Spiel mit lauter gleichen Auszahlungen (Binomialverteilung), Gleichstand-Zustand, ε = 1, Schärfe an den Reglergrenzen, Fixpunktzählung am Gitterpunkt 1/2 (Fehler der ersten Fassung der Vormessung) und der Plateau-Fall des Maximums bei p\* = 1/2.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README. **Verdikte** (`test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich je Fehlermodell, jede Spielklasse, die drei ▶️-Knöpfe einzeln, Worst-Case n = 1 000 mit 3·10⁶ Aktualisierungen), Permalink, PDF, Presets samt Hilfetext-Aussagen, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung** (`tools/oracle_mutation.py`, ganze Suite ohne Oberflächen-Tests und ohne `test_threshold_table.py`, ohne `ggw_rng.py`): alle **587** Mutationsstellen in `ggw_game`, `ggw_chain`, `ggw_limit`, `ggw_three`, `ggw_sim`, `ggw_experiments` und `ggw_evaluation` geprüft; **565** von der Suite gefangen, **22** als äquivalent festgehalten (`tests/mutation_state.json`: der konstante Summand lgamma(n + 1) kürzt sich in der Normierung; `>` gegen `>=` und `<` gegen `<=` an Stellen, die ein vorangehender Fall (a = d, b = c) oder die Wahrscheinlichkeit null für Gleichheit unerreichbar macht; Bisektionsschritte unterhalb der Gleitkommagenauigkeit; zusätzliche Zufallszahlen am Ende eines Stroms; eine Zahl der Logit-Gleichgewichte von 2 statt 1 oder 3), 0 Zeitüberschreitungen. Nicht mutiert: `ggw_scenario` (das Werkzeug lässt die Szenario-Erzeugung aus), Abbildungen, PDF, Layout, Presets, Konstanten und die wörtlich kopierte `ggw_rng.py`. Der erste Lauf (419 Stellen in fünf Modulen) ließ 38 Überlebende, der zweite (174 weitere Stellen in Auswertung und Messreihen) 29. Die Lücken waren: Grenzfälle der Bisektion (Gleichheit mit dem Niveau, Zahl der Schritte), Lücken mit Nachkommastellen in der Klassifikation, Randmaxima der Ratenfunktion, die Grenzen der Zeitformel, der genaue Zufallsstrom der beiden Simulationen (jetzt mit dem skalaren SplitMix64 nachgespielt), die Schwellen der Verdikte (`tests/test_gaps.py`, `test_verdicts.py`) und die Zählung der Drei-Formate-Auswahl (`tally_outcomes` als eigene Einheit). Die Eckfall-Matrix der Oberfläche fand außerdem einen Überlauf (Austrittszeit 10^4642 Aktualisierungen je Betrieb bei n = 1 000, x = 40, jenseits des Gleitkommabereichs) in der Zeitanzeige; behoben, Regressionstest in `test_verdicts.py`. „Mutation sauber“ gilt nur für diese sieben Module.

## Tests

321 Tests (`pytest tests`), rund 2 Minuten lokal; die langsamsten sind die beiden Messungen in `test_claims.py` (Suche über 2 400 Spiele mit drei Formaten, Simulation der Austrittszeit) und die AppTest-Läufe über alle Regler. Zusammensetzung: Kern gegen Handrechnung 85 (`test_game.py` 17, `test_chain.py` 26, `test_limit.py` 14, `test_three.py` 12, `test_sim.py` 16), Orakel 27, Claims 28, Verdikte und Formate 41, Randfälle und Mutationslücken 28 (`test_edge_cases.py` 16, `test_gaps.py` 12), Messreihen und Schwellentabelle 27, Szenario und SplitMix64-Kopie 11, Abbildungen und PDF 17, Oberfläche (AppTest, Layout, Presets, Footer, Laufzeit-Importe) 57.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht und sechs Tabs mit je einem Experiment |
| `ggw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ggw_game.py` | Klassen, p\*, Vorteil, Vorzeichenzählung, Barriere, Potential |
| `ggw_chain.py` | Raten, stationäre Verteilung, Massen, Schwelle x\* |
| `ggw_limit.py` | Ratenfunktion, Logit-Gleichgewichte, kritischer Wert |
| `ggw_three.py` | Kette über Zusammensetzungen, GTH |
| `ggw_sim.py` | Einzelspieler-Simulation, Austrittszeit |
| `ggw_scenario.py` | Zufallsspiele mit drei Formaten, Startzustand |
| `ggw_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo`) |
| `ggw_experiments.py` | Die Messreihen E1 bis E6, Schwellentabelle |
| `ggw_evaluation.py` | Einstellungen, Ansichten, Verdikte als reine Funktionen |
| `ggw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ggw_pdf_export.py` | PDF-Export (fpdf2) |
| `ggw_layout.py` | Fester Kopf (Stylesheet) |
| `ggw_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `data/threshold_table.json` | Vorgerechnete Schwellentabelle mit Erzeuger-Hash |
| `tools/make_threshold_table.py` | Erzeugt die Schwellentabelle neu |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnung, Orakel, Claims, Verdikte, Randfälle, Presets, Szenario und Kopie, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Netze und Nachbarschaften** und die **Wechselzeit der Konvention** auf ihnen (siehe `komplexe-ansteckung-demo` für Schwellen auf Netzen).
- **Gleichzeitige Aktualisierung aller Betriebe**, **zwei Rollen**, **heterogene Betriebe** (verschiedene Schärfe oder Fehlerraten).
- Drei Formate für n > 12 und andere Spielklassen als die in E5; mehr als drei Formate.
- **Menschenexperimente:** Eigennutz und Lernregeln sind Modellannahmen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Demo läuft mit Python 3.12 oder neuer (CI: 3.12). In der Streamlit Cloud ist die Python-Version in den App-Einstellungen gewählt. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
