# fpv-autocut

Automatický výběr nejlepších momentů a střih FPV záběrů.
[English](README.md) · **Česky**

Namiř ho na složku s lety. Nejlepší momenty najde z **dat gyra** (flipy, rolky,
střemhlavé lety, rychlé průlety) a z **pohybu v obraze** a vyrenderuje:

| Výstup | Výchozí | Poznámka |
|---|---|---|
| `reel.mp4` | 3 min, 16:9 | nejlepších ~10–15 % session, střihy zarovnané na doby hudby |
| `long.mp4` | 12 min, 16:9 | volnější výběr, delší souvislé úseky, původní zvuk |
| `short_01.mp4`, `short_02.mp4` | 30 s, 9:16 | Instagram / Facebook Reels; nejsilnější moment první |
| `*.fcpxml` | | jeden pro každý výstup (i shorty, s přerámováním); import do DaVinci Resolve |
| `plan.json`, `report.md`, `sheets/*.jpg` | | co bylo vybráno a proč; lze upravit a znovu vyrenderovat |

Vzniklo pro **RunCam Thumb Pro**, který vedle každého videa zapisuje gyro log
Gyroflow `.gcsv`, ale funguje s jakoukoli kamerou. Bez gyra se opírá o pohyb
v obraze a zvuk.

## Proč gyro

Z obrazu samotného nepoznáš power loop od roztřeseného visení. Z gyra ano:
`fpv-autocut` integruje rotaci po osách v okně 1,5 s. Vibrace a kmitání se
vyruší, zatímco flip dá zhruba 360°. Prostoje na zemi (gyro < 20 °/s a stojící
obraz) se zahazují automaticky.

## Instalace

Potřebuješ Python ≥ 3.11 a ffmpeg v `PATH`.

```powershell
winget install Gyan.FFmpeg          # Windows; macOS: brew install ffmpeg
git clone https://github.com/mareskav/fpv-autocut
cd fpv-autocut
uv sync                             # jádro: jen numpy, funguje na Windows ARM64
uv sync --extra gemini              # volitelně: hodnocení přes Gemini
uv run fpv-autocut --help           # nebo aktivuj .venv a volej fpv-autocut přímo
```

## Použití

```powershell
# vše: analýza, plán, render
fpv-autocut run D:\FPV\2026-10-24 -o D:\FPV\edit --music D:\music\track.mp3

# nejdřív rychlý náhled v poloviční velikosti
fpv-autocut run D:\FPV\2026-10-24 -o D:\FPV\edit --music track.mp3 --draft

# jen plán, podívej se na report.md a sheets\*.jpg, uprav plan.json a pak render
fpv-autocut plan D:\FPV\2026-10-24 -o D:\FPV\edit --sheets
fpv-autocut render D:\FPV\edit\plan.json --only reel,short_01
```

Užitečné volby: `--reel 150`, `--long 600`, `--shorts 3`, `--short-len 20`,
`--gemini`, `--config my.toml`, `--hwaccel auto`, `-j 4`.

| Volba | Účinek |
|---|---|
| `--no-beat-sync` | hudba zůstane, ale střihy se nezarovnávají na doby |
| `--gyro-offset 0.2` | výchozí odhad posunu gyro↔video (sekundy) |
| `--no-gyro-sync` | použije `--gyro-offset` tak, jak je, bez zarovnání korelací |
| `--no-fcpxml` | (`render`) nevytvoří soubory `.fcpxml` |

Kontaktní listy (jedna miniatura na segment, popořadě) jdou vytvořit i samostatně:

```powershell
fpv-autocut sheet D:\FPV\edit\plan.json
```

Gyro log se k videu zarovná automaticky; výsledek se zapíše jako
`gyro sync NAME: +x s (corr c)`. Slabé shody se ignorují.

Analýza se ukládá do `OUT/cache/`, takže přeplánování s jiným nastavením trvá
vteřiny.

### Gyroflow

Pokud vedle `foo.mp4` leží `foo_stabilized.mp4` (výchozí název výstupu
Gyroflow), použije se pro render. Analýza dál běží na originálu a jeho `.gcsv`.
Nejdřív stabilizuj, pak spusť `fpv-autocut`.

### DaVinci Resolve

*File → Import → Timeline…* a vyber `reel.fcpxml`. Klipy odkazují na tvoje
originály (nebo stabilizované soubory), takže můžeš dělat grading, přechody
a jemné ladění střihů v plné kvalitě. Hudba se naimportuje jako připojený zvukový
klip pod prvním záběrem. Přerámování shortů na 9:16 je transformace
„nejlepší snahou“, rámování si v Resolve zkontroluj. Pro 4:3 záběry na 16:9
časové ose nastav *Project Settings → Image Scaling → Mismatched resolution* na
*Scale full frame with crop*.

### Gemini (volitelně)

`--gemini` nahraje 360p, 5 fps tichou proxy každého letu se silnými kandidáty
a nechá Gemini ohodnotit až 60 momentů 1–10. Odměňuje blízkost překážek, mezery,
čisté flipy a scenérii, penalizuje záběry jen na oblohu a rozmazané. Vrací
i polohu hlavního objektu, která řídí ořez 9:16. Předem nastav `GEMINI_API_KEY`.
Model: `gemini_model` v konfiguraci (výchozí `gemini-flash-latest`). Hodnocení se
cachuje v `OUT/gemini/`. Bez Gemini se ořez 9:16 řídí středem pohybu v obraze
(`auto_crop`).

## Doporučené nastavení RunCam Thumb Pro

Podle [dokumentace Gyroflow](https://docs.gyroflow.xyz/app/getting-started/supported-cameras/runcam):

- **1440p @ 60 fps**: nativní 4:3. Ořez 9:16 zachová 42 % šířky, z 16:9 jen 32 %.
- EIS **vypnuto**, korekce zkreslení **vypnuta**.
- Závěrka 45–90° (1/480–1/240 s při 60 fps), ISO co nejnižší.

## Ladění

Všechny prahy jsou v [`fpv_autocut/config.py`](fpv_autocut/config.py) a dají se
přepsat TOML souborem, viz [`fpv-autocut.example.toml`](fpv-autocut.example.toml).
`timeline.csv` ukazuje všechny signály a výsledné skóre po 0,5 s, takže vidíš,
proč se něco vybralo, nebo ne.

## Práce s Claude Code

Repo obsahuje [`CLAUDE.md`](CLAUDE.md) a skill `/fpv-autocut` v `.claude/skills/`.
Otevři Claude Code v této složce a řekni např. *„ostříhej dnešní lety z
D:\FPV\2026-10-24 s touhle písničkou“*. Spustí plán, projde kontaktní listy,
upraví plán a vyrenderuje.

## Vývoj

```bash
uv sync                      # nainstaluje i dev skupinu (pytest, ruff)
uv run pytest                # unit testy + end-to-end test na syntetických záběrech
uv run ruff check .          # lint
uv run python tests/synth.py demo/   # syntetický let se známým flipem / rychlým průletem
```

## Licence

MIT
