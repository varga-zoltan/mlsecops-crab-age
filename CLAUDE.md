# CLAUDE.md — Crab Age MLOps (MLSecOps féléves feladat)

Ez a fájl a Claude Code állandó utasításkészlete ehhez a repóhoz. Minden munkamenet elején olvasd el.

## 0. Kivel dolgozol és hogyan

- A felhasználó **Varga Zoltán**, adattudomány MSc hallgató. **Kevés programozási tapasztalata van**, a git, a Docker és a CLI még új neki.
- **Magyarul kommunikálj.** A kód, a kommentek, a commit üzenetek és a README angolul íródnak (a tanár repói is angolok). Kivétel a README „AI használat” szakasza: az lehet magyarul is.
- **Lépésről lépésre haladj.** Egyszerre egy kis, ellenőrizhető feladatot csinálj. Minden lépés előtt 1–3 mondatban mondd el, **mit** csinálsz és **miért**, és kösd a tananyag fogalmaihoz (pl. „ez az elfogadási kapu, a jóság csökkenését akadályozza meg”).
- **Git-parancsot soha ne futtass magad** olyat, ami módosítja a repót (`git add`, `commit`, `push`, `merge`, `checkout -b`, `rebase`, `reset`). Írd ki a pontos parancsot egy kódblokkba, magyarázd el, mit csinál, és **Zoltán írja be**. Olvasó parancsot (`git status`, `git log`, `git diff`) futtathatsz.
- Ha valamit **grafikus felületen** kell csinálni (GitHub weboldal, Docker Desktop, böngésző, MLflow UI), számozott lépésekben írd le: hova kattintson, mit kell látnia.
- Minden kódváltozás után futtasd: `flake8`, `pyright`, `pytest` + coverage. Csak zöld állapotban javasolj commitot.
- A munkamenet végén foglald össze: mi készült el, mi a következő lépés, mi maradt nyitva.
- Ha egy döntés több irányba is mehet (pl. melyik modell, mit logoljunk hova), **kérdezz**, ne találgass.

## 1. A projekt

- **Adat:** Kaggle *Crab Age Prediction* (CC0), `data/raw/CrabAgePrediction.csv`, 3893 sor, 9 oszlop, nincs hiányzó érték.
  - Jellemzők: `Sex` (M/F/I, kategorikus), `Length`, `Diameter`, `Height`, `Weight`, `Shucked Weight`, `Viscera Weight`, `Shell Weight` (numerikus).
  - **Célváltozó:** `Age` (hónap, egész szám) → **regresszió**.
  - Ismert adatprobléma: kiugró és 0 értékek, pl. a `Height` max 2.825 a ~0.36-os medián mellett. Ezt az előfeldolgozás kezeli, és dokumentáljuk.
- **A modell jóságának konkrét értéke NEM értékelési szempont.** A cél (a feladatleírás szerint) kettős:
  1. **a jóság lehetőleg automatikus növelése**: minden tanításkor automatikus hiperparaméter-keresés és több modelltípus kipróbálása, a legjobb jelölt lesz a challenger (lásd 3. fejezet, `/train`);
  2. **a jóság csökkenésének automatikus megakadályozása**: elfogadási kapu + champion/challenger. Rosszabb modell soha nem kerül élesbe.
- A feladatleírás: `docs/assignment.md` (a tanár szövege, változtatás nélkül). Ha kétség van, ez az irányadó.
- Beadási határidő: 2026-12-06, de a cél a mielőbbi beadás.

## 2. Referenciák — ezeket kövesd mintaként

A tanár (dr. Fried Zoltán) példarepói a `../reference/` mappában vannak (NEM részei ennek a repónak):
`github-fastapi`, `github-ml-tests`, `github-docker-mlflow`, `github-docker`, `github-simple-actions`, `github-python-testing`.
Ha egy megoldásra van minta ezekben, **azt a stílust kövesd** (átalakítva, nem szó szerint másolva). Ahol a minta „churn” osztályozás, ott nálunk „crab age” regresszió van.

Fontos órai elvek (laborok + diák):
- `src/` layout, telepíthető csomag (`pip install -e .`), a logika modulokban van, nem notebookban.
- A **preprocessing a sklearn `Pipeline`-on BELÜL** van (nincs külön `scaler.pkl`), így nincs training/serving skew.
- **`pickle` = kódfuttatás**: modellt MLflow model formátumban, **signature-rel** mentünk és töltünk be.
- Konfiguráció `pydantic-settings`-szel (`extra="forbid"`, env prefix `CRAB_`), `.env.example` valódi titkok nélkül.
- **Nincs `print()` az `src/`-ben**, helyette `logging` (JSON formátum, correlation/request id).
- A tanítás szerződése: **artifact + exit code**. A `python -m crab_age.train` nem nulla kóddal lép ki, ha az elfogadási kapun elbukik.
- Lineage: a modell MLflow runja tartalmazza a `git_sha` és a `dvc_data_hash` tageket. Opcionálisan `docs/lineage/<git_sha>.json`.
- Titok soha nem kerül gitbe (docker `secrets/` fájlok + `.env`, mindkettő gitignore-olva, csak `*.example` megy be).

## 3. ⚠️ A LEGFONTOSABB KÖVETELMÉNY: egyetlen `docker compose up`

A tanár a repót leklónozza, és **csak ennyit ír be:** `docker compose up`. Ha ezután bármi kézi lépést igényel vagy hibára fut, az a legsúlyosabb hiba. Ezért:

- **Ne kelljen előtte `cp .env.example .env`, se `secrets/` fájlokat létrehozni.** A compose fájlban minden változónak legyen működő fejlesztői alapértéke (`${VAR:-default}`). A `.env` és a secrets csak opcionális felülírás. Ennek a kompromisszumnak a biztonsági oldalát (helyi dev jelszavak a compose-ban) a README röviden indokolja.
- **Friss klónon nincs adat és nincs modell.** A megoldás (Zoltánnal egyeztetve):
  - Az adat **DVC-ben van** (a feladat kötelezően előírja). A gitben csak a `.dvc` pointer és a `dvc.lock` van.
  - A **DVC remote egy publikus, csak olvasható tároló a GitHubon**: egy külön repó (`mlsecops-crab-age-data`), amit a DVC `http` remote-ként olvas (`https://raw.githubusercontent.com/varga-zoltan/mlsecops-crab-age-data/main`), így a tanár gépén is jelszó nélkül működik a `dvc pull`. Feltöltés: a repó helyi klónja local remote-ként (`dvc push -r upload`), utána git push abban a repóban. A dataset CC0 licencű, 250 KB, ezért publikussá tehető.
  - **Ezt az első napon, a DVC beállításakor kell kipróbálni:** friss mappában `dvc pull` jelszó nélkül. Ha nem működik, ne kerülgesd, hanem mutass alternatívát (pl. DagsHub DVC remote), és döntsünk együtt.
  - MinIO-t nem használunk (sem a DVC-hez, sem az MLflow-hoz): a hivatalos image már nem elérhető, a harmadik féltől származó, nem karbantartott image ellátási lánc kockázat. Az MLflow artifactok egy Docker volume-ra kerülnek.
  - Egy egyszeri **init konténer** a `docker compose up` alatt: `dvc pull` → ha még nincs `champion` modell, első tanítás (a `dvc repro` / train pipeline-nal) → kilép. Az `api` és a `dashboard` csak ennek sikeres befejezése után indul (`service_completed_successfully`). Így a `/predict` és a dashboard azonnal működik.
- **Indulási sorrend:** healthcheckek + `depends_on: condition: service_healthy / service_completed_successfully`. Semmi nem induljon, amíg a függősége nem kész.
- Az image-ek a compose-ból épüljenek (`build:`), ne kelljen előtte `docker build`.
- **Fix, ütközésmentes portok**, és a README-ben egy táblázat: melyik szolgáltatás melyik URL-en érhető el.
- **Ellenőrzés (kötelező minden nagyobb lépés után és beadás előtt):** új, üres mappába klónozás → `docker compose down -v` (a régi volume-ok törlése) → `docker compose up` → minden szolgáltatás `healthy`, a `/predict` válaszol, az MLflow-ban van modell, a Streamlit mutatja a riportot. Ezt Zoltánnal együtt végezzük, lépésenként.

## 3b. Architektúra

| Service | Szerep |
|---|---|
| `postgres` | MLflow backend store + külön `monitoring` adatbázis a predikciós log táblának |
| `init` | egyszeri: `dvc pull` a GitHub-os DVC remote-ról + első tanítás, ha nincs `champion` |
| `mlflow` | tracking server + model registry (`--serve-artifacts`, az artifactok egy Docker volume-on) |
| `api` | FastAPI: `/token`, `/health`, `/version`, **`/train`**, **`/predict`** |
| `dashboard` | Streamlit, benne az **Evidently** drift report (reference = training adat, current = logolt kérések) |
| `airflow` (opcionális, +5%, a végén) | ütemezett újratanítás és/vagy drift-ellenőrzés DAG |

Mintának a `github-fastapi/docker-compose.yaml`-t vedd (x-env anchor, healthcheck, `depends_on: condition: service_healthy`, docker secrets).
A konténerek egymást **service névvel** érik el (`http://mlflow:5000`, `postgres:5432`), soha nem `127.0.0.1`-gyel.
A gép **Apple Silicon (M3, arm64)**: ellenőrizd, hogy minden image-nek van-e arm64 változata.

### API követelmények
- Pydantic request/response modellek, mezőszintű validációval (`Sex: Literal["M", "F", "I"]`, méretek `> 0`, ésszerű felső korlát). Hibás input → 422.
- JWT auth (`OAuth2PasswordBearer`, Argon2 hash, `SECRET_KEY` env-ből) a `/train` és a `/predict` végponton, ahogy a `v5_middleware.py` mintában.
- Modell betöltése `lifespan`-ben a registryből: `models:/crab-age@champion`.
- **`/train`**: tanítás, MLflow-ba logolás (paraméterek, metrikák: MAE, RMSE, R²; adat, modell signature-rel), regisztráció.
  **Automatikus javítás:** a tanítás nem egyetlen fix modellt illeszt, hanem keresést végez. Több modelltípust próbál (pl. Ridge, RandomForest, GradientBoosting), mindegyikre hiperparaméter-keresést futtat (`RandomizedSearchCV` vagy `GridSearchCV`, a keresési tér a `params.yaml`-ban), és keresztvalidációval választja ki a legjobbat. Minden jelölt külön (nested) MLflow runba kerül, így a dashboardon látszik a javulás. Ugyanez fut le az ütemezett újratanításkor is (Airflow, vagy ha az nincs, a `/train` hívásakor), ha új adat érkezik.
  **Elfogadási kapu:** a challenger csak akkor kapja meg a `champion` aliast, ha nem rosszabb a jelenlegi championnál (és egy abszolút küszöbnél sem). Különben marad a régi modell, és ezt logoljuk.
- **`/predict`**: predikció. **Minden kérést és predikciót logolunk** (timestamp, request id, input mezők, predikció, modell verzió), ez az Evidently „current” adata. A predikciók MLflow artifactként is mentődnek (külön inference experimentbe, pl. batch-enként).
- A kérés/predikció logolást egy **paraméterezhető dekorátor** végzi (a `github-python-testing/src/case5/decorator_params.py` mintájára).
- Middleware: request id, időmérés, logolás.

### Streamlit + Evidently
- A Streamlit vékony réteg legyen: **az üzleti logika (report generálás, adatbetöltés) az `src/crab_age/monitoring/`-ban** van, és ott tesztelve.
- Az Evidently-t **a legegyszerűbb, legbiztosabb módon** használjuk: az aktuális stabil verziót, pontosan rögzítve (`==`), annak hivatalos dokumentációja szerint. Csak a minimum kell: `DataDriftPreset` riport (reference vs. current) → HTML → beágyazva a Streamlitbe, plusz a drift arány számként. Telepítés előtt ellenőrizd, hogy kompatibilis-e a többi rögzített csomaggal (pandas, numpy, Python 3.12). Ha nem, szólj, és döntsünk együtt. A tananyag régi (`evidently<0.7`) API-ját ne használd.

## 4. Kódminőség — számszerű követelmények (a tanár ellenőrzi!)

- **Coverage ≥ 95%** az `src/` teljes kódjára, a tesztkód is beleszámít (`coverage run -m pytest`, `coverage report -m --fail-under=95`).
- **flake8**, `max-line-length = 120`, a hibák max. 1%-a. Cél: 0.
- **Pyright** csak az `src/`-re, „szelíd strict” módban, **0 hiba**:
  ```json
  { "include": ["src"], "typeCheckingMode": "strict", "reportMissingTypeStubs": false,
    "reportUnknownMemberType": false, "reportUnknownArgumentType": false, "reportUnknownVariableType": false }
  ```
- **PEP8** betartása minden Python fájlban (src, tests, dashboard, airflow, scripts). A flake8 ezt ellenőrzi.
- **Type hint mindenhol** (függvény paraméterek és visszatérési értékek, fontos változók), a tesztekben és a vékony rétegekben is.
- `# noqa`, `# pragma: no cover`, `omit`, `# pyright: ignore[...]` **csak a sor melletti indoklással**, pl. `# pragma: no cover - entrypoint guard, tested via subprocess`.
- PEP8, docstringek a publikus függvényeken.
- `requirements.txt` (runtime) és `requirements_dev.txt` (dev + teszt), **pontos verziókkal** (`==`). A referencia repók verzióit vedd alapul (pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2 stb.), és ellenőrizd a kompatibilitást.
- Python 3.12 (Docker és CI).

### Tesztelés (a `github-python-testing` és a `github-ml-tests` stílusában)
- `tests/` a forrásstruktúrát tükrözi. Pozitív és negatív tesztek, `pytest.mark.parametrize`.
- **Biztonsági / negatív tesztek:** rossz típus, negatív és 0 méret, ismeretlen `Sex`, hiányzó mező, túl nagy payload, auth nélküli hívás → 401.
- **Hypothesis** property-based teszt (pl. bármely érvényes inputra a predikció véges, pozitív szám).
- **Test double-ök**: az MLflow-t és a külső szolgáltatásokat mockoljuk (`pytest-mock`, `create_autospec`), így a tesztek **MLflow szerver és Docker nélkül** futnak a CI-ban.
- FastAPI: `TestClient` + `app.dependency_overrides` az auth stubolására, a végén `clear()`.

## 5. Git szabályok

- **Trunk-based:** rövid életű feature branchek (`feature/<téma>`), **Pull Request + merge commit** (NEM squash, NEM rebase), így megmarad a feature branch history. Min. 2 feature branch, valójában minden nagyobb részhez egy.
- A `main`-re **csak merge útján** kerül commit (kivétel: typo javítás). A `main` branch védelmét Zoltán állítja be a GitHubon, ehhez adj lépésenkénti útmutatót.
- **Atomic commit:** egy commit = egy döntés. Gyakori, kis commitok, összesen **legalább 50**.
- **Commit üzenet formátuma** (a tárgyalt formátum, version_control diák):
  ```
  keyword: one short line summary (max 80 char)

  Longer description: what changed and why?

  Signed-off-by: Zoltan Varga <email>
  ```
  - `keyword` = téma vagy modul, kisbetűvel: `data:`, `dvc:`, `train:`, `api:`, `test:`, `docker:`, `ci:`, `mlflow:`, `monitoring:`, `doc:`, `req:`, `fix:`, `pre-commit:`.
  - A `Signed-off-by` sort a `git commit -s` adja hozzá automatikusan, ezért **mindig `-s`-sel commitolunk**.
  - Mivel a commit üzenet többsoros, a javasolt parancs formája: `git commit -s -m "keyword: summary" -m "Longer description..."`.
- Gitbe **SOHA** nem kerül: nyers vagy feldolgozott adat (csak `.dvc` pointer), modellfájl, `mlruns/`, `*.db`, `.env`, `secrets/`, `.venv/`, cache-ek, notebook output.
- Adat: **DVC**, remote = publikus GitHub repó (lásd 3. fejezet). Szabály: **`dvc push` minden `git push` mellé**, ha az adat változott.

## 6. Automatizálás

- **pre-commit** (`.pre-commit-config.yaml`, a Lab 3 mintájára, de ruff helyett **flake8** és **pyright**):
  `check-added-large-files --maxkb=1024`, `check-merge-conflict`, `check-yaml`, `check-json`, `end-of-file-fixer`, `trailing-whitespace`, `detect-private-key`, `gitleaks`, `nbstripout`, `flake8`, `pyright` (local hook).
- **GitHub Actions** = a feladatleírásban szereplő „automatikusan futó pre-commit hookok” a **tesztekre, a coverage-re és a docker image készítésére**. Ez kötelező, és minden PR-on zöldnek kell lennie merge előtt. A `github-fastapi/.github/workflows/docker.yaml` mintájára, push-ra és PR-ra a `main` felé:
  1. `lint`: flake8 + pyright
  2. `test`: pytest + `coverage report --fail-under=95`
  3. `build`: Docker image build (+ push GHCR-be a `main`-en), `needs: [lint, test]`
  4. opcionális: `hadolint`, `trivy` image scan, `gitleaks`
- **Dockerfile** a tananyag szerint: pontos base image tag, előbb `requirements.txt`, utána a kód, `PIP_NO_CACHE_DIR=1`, **nem root user** (`USER`), `HEALTHCHECK`, exec form `ENTRYPOINT`/`CMD`, `.dockerignore`.

## 7. Javasolt mappastruktúra

```
mlsecops-crab-age/
├── .github/workflows/ci.yaml
├── .pre-commit-config.yaml  .flake8  pyproject.toml  pyrightconfig.json
├── requirements.txt  requirements_dev.txt  .env.example  .dockerignore  .gitignore
├── docker-compose.yaml  Dockerfile  secrets_example/
├── dvc.yaml  params.yaml  dvc.lock  data/raw/*.dvc
├── src/crab_age/
│   ├── config.py  logging_setup.py
│   ├── data/ (load, validate, prepare)
│   ├── model/ (pipeline, train, evaluate, promote)
│   ├── api/ (main, schemas, auth, logging decorator)
│   └── monitoring/ (prediction log, drift report)
├── dashboard/app.py          # vékony Streamlit réteg
├── airflow/dags/             # opcionális, vékony réteg
├── tests/
├── metrics/metrics.json
└── docs/ (assignment.md, screenshots/, lineage/, ai-usage.md)
```

## 8. Dokumentáció és beadás

- **A README stílusa:** úgy írd, mintha Zoltán írta volna. Tömör, lényegre törő, professzionális, de egyszerű nyelvezetű. Rövid mondatok, nincs marketingszöveg, nincs túlzó jelző, nincs felesleges emoji, nincsenek hosszú bevezetők. Inkább rövid listák, táblázatok és működő parancsok. Ha valami nem működik vagy technikai adósság, azt egyenesen írd le.
- Az „AI használat” szakasz ettől függetlenül **őszintén** leírja, mire és hogyan használtuk az MI-t.

- A `README.md` (markdown) tartalmazza: áttekintés + architektúra ábra (mermaid), telepítés, `docker compose up`, a végpontok használata (curl példák), az MLflow/Streamlit/Airflow URL-jei, a tesztek és minőségi eszközök futtatása, az adat előfeldolgozás leírása, és egy **„AI használat”** szakasz (mire és hogyan használtuk a Claude-ot / Claude Code-ot).
- **Screenshotok** a `docs/screenshots/` mappába (gitbe commitolva, beszédes fájlnévvel, pl. `01_mlflow_experiments.png`). A README hivatkozza és be is ágyazza őket. Ha eljutunk ide, adj Zoltánnak lépésenkénti útmutatót, mit nyisson meg és mit fotózzon. Ellenőrzőlista:
  - [ ] **MLflow dashboard**: az experiment a runokkal és metrikákkal (látszik az automatikus javulás), plusz a model registry a `champion` aliasszal
  - [ ] **REST API működése és tesztje**: Swagger UI (`/docs`) sikeres `/token` → `/train` → `/predict` hívással és válasszal, valamint a hibás inputra adott 422-es és az auth nélküli 401-es válasz
  - [ ] **Docker konténerek működése**: `docker compose ps` kimenet (minden `healthy`) és/vagy a Docker Desktop konténerlistája
  - [ ] **Streamlit dashboard**, rajta az **Evidently drift report**
  - [ ] **Airflow dashboard** a DAG-gal és egy sikeres futással (ha van Airflow)
  - [ ] Ajánlott pluszként: zöld **GitHub Actions** futás, és a PR-ok/merge-ek listája a GitHubon
- Az API tesztelése nem csak screenshot. Legyenek **automatikus pytest tesztek** is a végpontokra (`TestClient`), és egy egyszerű **Python smoke-test szkript** (`scripts/smoke_test.py`), ami a futó rendszer ellen végighívja a `/token` → `/train` → `/predict` láncot. A kimenete is mehet screenshotra.
- A beadandó ZIP-hez kell az **előfeldolgozott dataset fájl** is, ezt a `prepare` stage állítja elő.

## 9. Tilos / óvatosan

- Ne írj felül a `data/raw/` alatt semmit.
- Ne tegyél titkot kódba, `docker-compose.yaml`-ba vagy `Dockerfile`-ba.
- Ne vegyél fel függőséget pontos verzió nélkül.
- Ne kapcsolj ki ellenőrzést (flake8, pyright, coverage) csak azért, hogy átmenjen. Javítsd a kódot, vagy indokold.
- Ékezetes karakter ne legyen a projekt elérési útjában (elrontja a `pip install -e .`-t).
