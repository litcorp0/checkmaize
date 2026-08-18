# CheckMaize — Complete Step-by-Step Guide (Zero Coding Experience Required)

This guide takes you from "I have a computer and a phone" to "the CheckMaize app
is installed on the phone and diagnosing maize leaves."

It assumes you have **never written code before**. Every click, every button, and
every expected result is described. If something goes wrong, jump to
[Troubleshooting](#troubleshooting).

---

## The Big Picture (read this first)

The project has two halves:

1. **The brain** — a machine-learning model that can look at a maize leaf photo
   and say which disease it has. Training this brain needs a powerful computer,
   so we use **Google Colab** — a free website where Google lends you a fast
   computer with a graphics card (GPU) that runs in your browser.
2. **The body** — the CheckMaize phone app (Expo/React Native). It holds the
   trained brain (as a small file called `model_int8.onnx`) and runs it
   **completely offline** on the phone.

The workflow in one sentence: **on Colab we download pictures of diseased maize
leaves → teach the brain → shrink the brain into a small file → copy that file
into the app → build the app → install it on a phone.**

```
  Colab (browser)                          Your computer                 Phone
  ----------------                         -------------                 -----
  Notebook 01: build picture sets ──►  data/manifests (CSV files)
  Notebook 02: train 5 brains     ──►  pick the best one
  Notebook 03: shrink the brain   ──►  app/assets/model/model_int8.onnx ──► APK install
```

**Time needed:** about 4–6 hours total, mostly waiting for downloads and training.

**What you need:**
- VS Code with the official **Colab extension** (recommended way to run the
  notebooks — installed from the VS Code Extensions marketplace: search
  "Colab" and pick the one by Google)
- A free Google account (to sign into the Colab extension)
- A free GitHub account (so the cloud runtime can read your project code)
- An Android phone (Android 7 or newer) with a USB cable
- A free Kaggle account (Google login — to download the Ghana dataset)
- (Optional, fallback only) a browser — Chrome or Firefox — to run the
  notebooks at colab.research.google.com instead of VS Code

---

## Part 0 — Put the project on GitHub (one-time, ~15 minutes)

Colab cannot see the files on your computer. It can, however, download (clone)
a project from GitHub. So we first upload the project to GitHub.

1. Go to <https://github.com> and sign up / sign in.
2. Click the green **New** button (top left), or the **+** button in the top
   right corner → **New repository**.
3. Repository name: `checkmaize`. Choose **Public** — this is important,
   because the Colab cloud computer has no way to log into your GitHub
   account, and it can only download from public repos. Don't worry about
   privacy: the big/private files (all the pictures, the trained brains,
   `node_modules`) are excluded by the project's `.gitignore`, so GitHub only
   ever sees the code and small list files. Do **not** tick "Add a README" or
   any other box. Click **Create repository**.
4. GitHub shows a page of commands. Copy the **"…or push an existing repository
   from the command line"** block. It looks like:

   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/checkmaize.git
   git branch -M main
   git push -u origin main
   ```

5. On your computer, open a terminal **inside the `checkmaize` project folder**
   and paste those two commands, one at a time, pressing Enter after each.
   When GitHub asks for credentials, use a **Personal Access Token** (not your
   password): GitHub → click your avatar → Settings → Developer settings →
   Personal access tokens → Generate new token (classic) → tick `repo` →
   Generate → copy the token → paste it as the password.
6. Refresh the GitHub page. You should see all the project files listed. Done —
   Colab can now reach your code.

> **If you created the repo as Private earlier:** the cloud computer will not
> be able to download it. Change it to Public (GitHub → the repo → **Settings
> → General → Danger Zone → Change repository visibility → Make public**),
> then re-run Cell 1 — it clones the repo automatically now.

> **No terminal on your computer?** Skip Part 0. In every Colab notebook below,
> use the "upload a zip" option shown in Cell 1 instead. Zip the whole
> `checkmaize` folder (right-click → Compress), upload it in the Colab files
> pane, and follow the OPTION B instructions printed by the notebook.

---

## Part 0.5 — Connect Colab to VS Code (one-time setup, ~10 minutes)

You don't need the Colab website. The official **Colab extension for VS Code**
runs the notebooks on Google's cloud computers (with the free GPU) while you
type and run them from VS Code.

1. In VS Code, open the **Extensions** view (the four-squares icon in the left
   toolbar, or press `Ctrl+Shift+X`).
2. Search for **Colab** and install the extension published by **Google**
   (its ID is `google.colab`).
3. Click the new **Colab icon** that appears in the left toolbar.
4. Click **Sign in** and use your Google account.
5. Click **Connect to a Colab runtime**. When asked which runtime, choose
   **T4 GPU** (free) — this makes training ~50x faster than your laptop.
6. Once connected, the extension's **file explorer** shows the cloud
   computer's file system (its `/content` folder). Files you drag INTO the
   explorer are uploaded to the cloud; files you drag OUT are downloaded to
   your computer.

> The cloud computer is temporary: Google wipes it when you disconnect or it
> times out. Anything important must be dragged back to your computer (the
> notebooks do this for you at the end of each part).

---

## Part 0.7 — Browser Colab quick path (one file, everything)

If you prefer the Colab website (colab.research.google.com) instead of VS Code,
use the all-in-one notebook: **`colab/00_full_run.ipynb`**. One upload, one
session, everything in order:

1. Go to **colab.research.google.com** → **File → Upload notebook** →
   pick `colab/00_full_run.ipynb` from your project folder.
2. **Runtime → Change runtime type → T4 GPU → Save**, then connect.
3. Run the cells **in order, top to bottom**:
   - Cells 1–5: data (PlantVillage + Ghana + splits) — ~45 minutes.
   - Cell 6: mounts your Google Drive and restores any earlier progress (browser only).
   - Cell 7: the big training run (~2–3 hours — keep the tab open).
   - Cells 8–14: scoreboard, winner, export, and the downloads.
4. At the end you download four zips: `splits.zip`, `report.zip`, `runs.zip`,
   and `artifacts.zip` + `fixtures.zip`. Save them for the assistant to commit.

**Why one file?** The Colab website gives every notebook its own temporary
computer — two notebooks cannot share the downloaded pictures. One file keeps
everything on one computer from start to finish.

**If the session drops mid-training:** reconnect, re-run Cells 1–5 (~45
minutes), then the training cell. Finished models are restored from your
Google Drive (mounted by Cell 6) and are skipped, so you lose almost nothing.

VS Code users: keep using the three notebooks (Parts 1–3) — the extension
shares one cloud computer between notebooks, so this is not needed.

---

## Part 1 — Build the picture sets (Colab Notebook 01, ~45 minutes)

This step gathers ~6,500 maize leaf pictures:
- **PlantVillage** (~4,000 pictures, photographed in labs) — downloaded
  automatically by the notebook from the Hugging Face website.
- **CCMT Ghana dataset** (~2,500 pictures of maize photographed on real farms
  in Ghana by researchers at the University of Energy and Natural Resources,
  Sunyani) — you download this yourself from the Kaggle website (free account).

Then the notebook sorts them into "training pictures", "checking pictures", and
"exam pictures" — the model learns from the training pictures and is graded on
the exam pictures it has never seen.

### Step 1.1 — Download the Ghana dataset to your computer

The Ghana dataset is the "CCMT" collection: maize leaves photographed on real
farms around Sunyani, Ghana, by the University of Energy and Natural Resources,
and checked by plant experts.

**Easiest route — Kaggle (recommended):**

1. In your browser go to:
   <https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection>
   (This is the same dataset; its own description cites the original Mendeley
   DOI `10.17632/bwh3zbpkpv.1` as the source.)
2. If Kaggle asks you to sign in, use your Google account (free).
3. Click the **Download** button (top-right of the dataset page).
4. Save the file (~1.3 GB) in your **Downloads** folder — it may be called
   `crop-pest-and-disease-detection.zip` or just `archive.zip`; both are fine.
   **Remember where it went.**

> **Fallback route — Mendeley (only if Kaggle is unavailable):** go to
> <https://data.mendeley.com/datasets/bwh3zbpkpv/1>, sign in (Google works),
> and use "Download All Files" (about 8 GB — it includes an extra augmented
> copy we don't need, so prefer Kaggle). If you use this route, rename the
> downloaded file to `Raw Data.zip`.

### Step 1.2 — Open the notebook in VS Code

1. In VS Code open the `checkmaize` project folder
   (**File → Open Folder**, choose `checkmaize`).
2. In the file list on the left, find `colab/01_dataset.ipynb` and click it.
3. The notebook opens as a list of grey boxes ("cells"). Make sure the Colab
   extension is connected (Colab icon → connected to a runtime).
4. You run cells one by one with the **play button (▶)** on the left of each
   cell (or `Shift+Enter`). **Wait for each cell to finish before running the
   next one.**

> **Browser fallback:** open <https://colab.research.google.com>, menu
> **File → Upload notebook**, choose the same file. Everything below works the
> same, except uploads use the left-side folder pane and downloads go to your
> browser's Downloads folder.

### Step 1.3 — Run Cell 1 (connect to your code)

Press ▶ on Cell 1. It checks where your project code is and installs any
missing tools.

- If the cell prints `Repo not on this runtime yet. Cloning from GitHub...`:
  it downloads the project itself — wait a few seconds.
- If it prints `Automatic clone failed`: the repo is still **Private** on
  GitHub (make it Public) or the runtime has no internet. Fix and re-run Cell 1.
- Otherwise it prints `Working in: ...` and `dependencies OK` — good.

Expected result: no red error text. (Red = problem; see Troubleshooting.)

> When the runtime wipes (new session), re-run Cell 1 first — it restores the
> repo by `git pull` if the folder is still there, or tells you to re-upload.

### Step 1.4 — Run Cell 2 (download PlantVillage, automatic)

Press ▶ on Cell 2. This downloads ~2 GB of images. **Expect 10–30 minutes.**

Expected result, at the bottom of the cell:

```
downloading data.zip (~2 GB, 10-30 minutes)...
downloaded: /root/.cache/huggingface/hub/...
plantvillage extraction done: 3852 maize images copied (0 missing)
```

(The cell downloads the color-image train+test lists directly from Hugging
Face — no `datasets` library involved, which avoids version problems.)

### Step 1.5 — Get the Ghana dataset onto the cloud and run Cell 3

Cell 3 has two ways to get the dataset onto the cloud machine.

> **VS Code: use OPTION B** — the OPTION A file-picker only works in the
> browser version of Colab, not in the VS Code extension.

**OPTION A (browser Colab only):** press ▶ on Cell 3, click the file-picker
button, choose the zip you downloaded in Step 1.1, wait for the upload bar,
and the rest of the cell runs by itself.

**OPTION B (the cloud downloads it directly from Kaggle — nothing to upload):**

1. On kaggle.com click your avatar → **Settings → API → Create New Token**.
   A file called `kaggle.json` downloads — open it in any text editor.
2. In Cell 3, find the OPTION B block. Paste the two values into the
   `KAGGLE_USERNAME` and `KAGGLE_KEY` lines, then delete the `#` at the start
   of those six lines.
3. Press ▶ on Cell 3. It downloads and extracts ~1.3 GB on the cloud machine
   (fast — no upload needed), then builds the dataset.

Expected result — three lines (numbers should match exactly):

```
Using: archive.zip (or crop-pest-and-disease-detection.zip)
leaf blight 1006
leaf spot 1259
healthy 208
ccmt_ghana ready: {'Leaf blight': 1006, 'Leaf spot': 1259, 'Healthy': 208}
```

(If you see `No Ghana dataset found on the cloud machine yet` — the upload
was not finished or OPTION B did not run. Re-check the cell and re-run it.)

### Step 1.6 — Run Cell 4 (build the splits + run the tests)

Press ▶ on Cell 4.

Expected result (numbers may vary slightly):

```
wrote 6500 rows to data/manifests/raw.csv
  common_rust: 1192
  gray_leaf_spot: 1772
  healthy: 1370
  northern_leaf_blight: 1991
train: ... rows
val: ... rows
test: ... rows
domain_shift_train: ... rows
...
... passed
```

The important part: **all tests say `passed`**, and **nothing is red**.

> **What just happened (in plain words):** the notebook wrote six lists (CSV
> files) that say exactly which pictures to use for learning and which for the
> exam. The tests prove that no exam picture was leaked into the learning set —
> that would be like giving a student the answers before the exam.

### Step 1.7 — Run Cell 5 (get the result onto your computer)

Press ▶ on Cell 5. It zips the six list files into `splits.zip`.

- **VS Code:** the cell prints where the file is (`/content/splits.zip`). Drag
  it from the Colab extension's file explorer onto your computer.
- **Browser Colab:** the download starts automatically to your Downloads folder.

### Step 1.8 — Put the lists into your project

1. Unzip `splits.zip` (right-click → Extract All).
2. Inside are six CSV files (`train.csv`, `test.csv`, `val.csv`,
   `domain_shift_*.csv`). Copy **all of them** into your project folder:
   `checkmaize/data/manifests/` (create that folder if it does not exist).
3. Commit them so Colab can see them later (only if you used GitHub):
   in a terminal inside the project folder run:

   ```bash
   git add data/manifests && git commit -m "data: add split manifests" && git push
   ```

Part 1 is done. ✅

---

## Part 2 — Train the brain and pick the best one (Colab Notebook 02, ~2–3 hours)

This step teaches **five different candidate brains** on the same pictures
(called "models" — think of them as five students with different study
habits), then shows you a scoreboard so we can pick the best student for the
phone. Two are small and fast (good for phones), the others are heavier.

### Step 2.1 — Make sure you are on the free GPU runtime

1. Open `colab/02_train_benchmark.ipynb` in VS Code (same as Step 1.2).
2. **Important:** click the **Colab icon** in the left toolbar and check you are
   connected to a **T4 GPU** runtime (the runtime name shows next to the
   connection). If not: disconnect, click **Connect to a Colab runtime** again,
   and choose **T4 GPU**. Without the GPU, training takes days instead of hours.

> **Browser fallback:** Menu **Runtime → Change runtime type → Hardware
> accelerator → T4 GPU → Save**.

### Step 2.2 — Run Cell 1 (connect + check the GPU)

Run Cell 1 (same options as Notebook 01). It now prints the GPU check itself:

```
torch: 2.x.x | GPU available: True
```

Expected: `True`. If it prints `False` (or a warning), you connected to a CPU
runtime — disconnect in the Colab extension and reconnect choosing **T4 GPU**
(see Step 2.1), then re-run Cell 1.

### Step 2.3 — Run Cell 2 (quick self-test of the code)

Run Cell 2. Expected: all tests `passed`. (This re-downloads nothing big; it
reuses the committed lists. It also re-extracts the picture files — Cell 1 of
Notebook 01's downloads do **not** carry over between notebooks; Colab forgets
everything when a notebook closes. If Cell 2 complains it cannot find
`data/raw/...`, re-run Notebook 01's Cells 2 and 3 first.)

### Step 2.4 — Run Cell 3 (the big training run)

Run Cell 3. **This is the long one — 2 to 3 hours.** The cloud runtime may
disconnect if you stay idle; keep VS Code open and connected, and click in the
notebook occasionally. If it stops, reconnect, re-run Cell 1, then re-run Cell
3 — finished models are skipped... (no — each model trains from scratch; if it
stops mid-model, re-running the cell restarts that model).

While it runs you will see lines like:

```
=== training custom_cnn ===
epoch 1/15 val_acc=0.4312
epoch 2/15 val_acc=0.6011
...
=== training mobilenet_v3_small ===
...
```

`val_acc` is the score on the checking pictures — it should climb towards
0.95–0.99.

### Step 2.5 — Read the scoreboard and pick the winner

When all five finish, the notebook prints a table like:

```
| model               | accuracy | macro_f1 | params    | onnx_bytes |
|---------------------|----------|----------|-----------|------------|
| custom_cnn          | 0.8123   | 0.8031   | 1_500_000 | 18_000_000 |
| mobilenet_v3_small  | 0.9611   | 0.9589   | 2_000_000 | 10_000_000 |
| mobilenet_v3_large  | 0.9688   | 0.9660   | 4_000_000 | 16_000_000 |
| efficientnet_b0     | 0.9722   | 0.9701   | 4_000_000 | 16_000_000 |
| resnet18            | 0.9655   | 0.9620   | 11_000_000| 44_000_000 |
```

- `accuracy` = share of exam pictures classified correctly. Bigger is better.
- `onnx_bytes` = how heavy the brain file is. Smaller is better for phones.

**How to pick:** choose the model with the highest accuracy whose file size is
reasonable for a phone (roughly under 20 MB). Usually `efficientnet_b0` wins,
or `mobilenet_v3_small` if you want a lighter app. **Write down the winner's
name** (e.g. `efficientnet_b0`) — you will type it in Part 3.

### Step 2.6 — Run Cell 4 (get the scoreboard + the trained brains)

Run Cell 4. It zips `report.zip` and `runs.zip` into `/content`.

- **VS Code:** drag both zips from the Colab extension's file explorer onto
  your computer.
- **Browser Colab:** the downloads start automatically.

Then:
- Unzip `report.zip` into `checkmaize/benchmarks/report/`.
- Unzip `runs.zip` into `checkmaize/artifacts/runs/`.
- Commit (GitHub users):

  ```bash
  git add benchmarks/report && git commit -m "data: add RQ3 comparison report" && git push
  ```

Part 2 is done. ✅

---

## Part 3 — Shrink the brain and hand it to the app (Colab Notebook 03, ~30 minutes)

The trained brain is too fat for a phone (16+ MB and slow). This step squeezes
it (a process called *quantization* — like compressing a photo into a smaller
file with barely any visible loss), checks the squeezed version still passes
the exam, and produces the exact files the app needs.

### Step 3.1 — Open Notebook 03

In VS Code open `colab/03_export.ipynb` from the project folder (same as
Step 1.2). Make sure the Colab runtime is connected.

### Step 3.2 — Run Cell 1 (connect + install the small tools)

Same as before — it finds the repo, installs anything missing, and prints the
`torch` version. No red text expected.

### Step 3.3 — Run Cell 2 (tell it which model won)

**Important:** in Cell 2 there is a line:

```python
winner = 'efficientnet_b0'  # SET THIS to the model chosen from the Task 8 comparison table
```

Change `'efficientnet_b0'` to the name you wrote down in Step 2.5 (one of:
`custom_cnn`, `mobilenet_v3_small`, `mobilenet_v3_large`, `efficientnet_b0`,
`resnet18`). Then run the cell. Expected: `winner: <your model>` with no red.

### Step 3.4 — Run Cell 3 (shrink + exam the squeezed brain)

Run Cell 3. Expected output is a small report that includes:

```
"parity_pass": true,
"fp32_accuracy": 0.97...,
"int8_accuracy": 0.96...,
"delta": -0.00...,
"ship_int8": true
```

- `parity_pass: true` = the squeezing did not damage the brain.
- `ship_int8: true` = the squeezed brain passed the exam (dropped less than 1%).
- If `ship_int8` is `false`: tell the researcher — we ship the bigger fp32 file
  instead (the notebook prints a warning about this).

### Step 3.5 — Run Cell 4 (package the app metadata)

Run Cell 4. It writes two small files (`labels.json` = the list of diseases in
order, `metrics.json` = the scoreboard row of the winner). Expected: a printed
JSON block with `model`, `test_accuracy`, etc.

### Step 3.6 — Run Cell 5 (get everything onto your computer)

Run Cell 5. It produces:
- `artifacts.zip` — contains `model_int8.onnx` (the squeezed brain), `labels.json`, `metrics.json`
- `fixtures.zip` — the parity-test pictures the app uses to double-check its math
- `onnx-contract.md` — the technical agreement between Python and the app

- **VS Code:** the cell prints where the zips are. Drag them from the Colab
  extension's file explorer onto your computer.
- **Browser Colab:** the downloads start automatically.

### Step 3.7 — Copy the files into the project

On your computer:

1. Unzip `artifacts.zip`. Copy these three files into
   `checkmaize/app/assets/model/`:
   - `model_int8.onnx`
   - `labels.json`
   - `metrics.json`
   (Overwrite the README.txt file that explains what belongs there — no,
   keep the README, just add the three files next to it.)
2. Unzip `fixtures.zip`. Copy the two files inside
   (`sample.png` and `reference_tensor.json`) into
   `checkmaize/app/src/ml/__tests__/fixtures/`, **replacing** the placeholder
   files that are already there. (These are the same names — overwrite them.)
3. Move `onnx-contract.md` into `checkmaize/docs/`.
4. Commit (GitHub users):

   ```bash
   git add app/assets/model app/src/ml/__tests__/fixtures docs/onnx-contract.md
   git commit -m "feat: bundle trained model and fixtures" && git push
   ```

Part 3 is done. ✅ The app folder now contains a real trained brain.

---

## Part 4 — Run the app tests and install it on your phone (~30–60 minutes)

### Step 4.1 — Install the app's dependencies (one time)

Open a terminal inside the **`app`** folder (`checkmaize/app`) and run:

```bash
npm install
```

This downloads the app's building blocks (first time: 5–10 minutes). During
the install you may see a message printed twice:

```
fix-onnxruntime: removed unimodule.json (unblocks Expo autolinking)
```

That is a **good sign** — it is our automatic repair for a known bug in the
onnxruntime package. Nothing to do.

### Step 4.2 — Check the app's math (the parity test)

In the same terminal run:

```bash
npm test
```

Expected:

```
Test Suites: 4 passed, 4 total
Tests:       11 passed, 11 total
```

**The parity test is the most important test in the project.** It proves the
app's photo-processing math produces exactly the same numbers as the Python
code that trained the brain. If this test ever fails after you replace the
fixture files, the model and the app disagree about how to read a photo — tell
the researcher before doing anything else.

### Step 4.3 — Type-check

```bash
npx tsc --noEmit
```

Expected: it prints **nothing** and returns to the prompt. (Red text = a code
problem — send it to the researcher.)

### Step 4.4 — Put the app on your phone (Option A: EAS cloud build, easiest)

EAS is Expo's free cloud service that builds the installable app for you.

1. Create a free Expo account at <https://expo.dev/signup>.
2. In the terminal (still in the `app` folder):

   ```bash
   npx eas-cli@latest login
   ```

   Enter your Expo username and password.

3. Start the build:

   ```bash
   npx eas-cli@latest build -p android --profile preview
   ```

   Answer the prompts: allow it to configure the project (yes), choose
   "preview" profile. EAS then uploads the code, builds in the cloud
   (10–20 minutes), and prints a link like
   `https://expo.dev/artifacts/eas/...apk`.
4. Open that link **on your phone's browser**, download the `.apk` file, tap it
   and allow "install from unknown sources" when Android asks.
5. Open **CheckMaize** from your app drawer. ✅

### Step 4.5 — Option B: build locally with USB debugging

If EAS is not available to you:

1. Put the phone in **Developer Mode** (Settings → About phone → tap "Build
   number" seven times), then enable **USB debugging** in Developer options.
2. Connect the phone by USB and accept the debugging prompt on the phone.
3. In the terminal:

   ```bash
   npx expo run:android
   ```

   The first build downloads a lot (30+ minutes). When finished, the app
   launches on the phone automatically.

### Step 4.6 — Try it

1. Open the app. Grant camera permission (and later location, if you want
   scans to be geotagged).
2. Point the camera at a maize leaf with clear symptoms, fill the frame, and
   press **Capture**.
3. After a second you should see a **Result screen**: the disease name, a
   confidence bar, and a button "What to do about it" with management advice.
4. If the app says **"Unclear photo..."** — the model was not confident enough.
   Take the photo again with better light or a closer frame. This honesty is
   deliberate: the app prefers to say "I don't know" over giving a wrong
   answer.
5. The **History** tab stores every scan (with GPS if allowed).
6. The **Contribute** tab saves unlabelled field photos and shares them (via
   WhatsApp/email) to you for future training — this is how the dataset grows.
7. The **About** tab shows the model's exam scores.

---

## Part 5 — The long-term loop (what to do over months)

1. **Field pilot:** have farmers/extension agents use the app; an expert labels
   the same leaves independently. Compare (see `docs/pilot-protocol.md`).
2. **Collect disagreements:** every photo where app and expert disagree is
   gold. Save them (Contribute tab).
3. **Re-train:** put the newly labelled photos into
   `data/raw/local/<disease>/...`, re-run Notebooks 01→02→03, and replace the
   three files in `app/assets/model/`. Rebuild the app. That is the whole
   upgrade cycle — no other changes needed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Colab says `No module named 'data'` or `ModuleNotFoundError` | You ran a cell before connecting to the repo. Run Cell 1 first; it must succeed without red text. |
| Cell 1 prints `Automatic clone failed` | Your GitHub repo is still **Private** (make it Public: Settings → General → Danger Zone → Change visibility) or the runtime has no internet. Fix, then re-run Cell 1. |
| `NameError: name 'REPO' is not defined` | You skipped Cell 1 or the kernel restarted. Just run Cell 1 once, then re-run your cell — the other cells now find the repo by themselves. |
| Notebook 02 Cell 2 cannot find `data/raw/...` | The cloud computer forgets downloads between notebooks. Re-run Notebook 01 Cells 2 and 3, then continue. |
| Cell 1 prints `GPU available: False` | In VS Code: Colab icon → disconnect → connect again choosing **T4 GPU**. Browser: Runtime → Change runtime type → T4 GPU. Then re-run Cell 1. |
| Runtime disconnects mid-training | Reconnect, re-run Cell 1, then re-run Cell 3 — it restarts the interrupted model. In the browser all-in-one notebook: re-run Cells 1–5, then Cell 7 — finished models are restored from Google Drive and skipped. |
| `git pull` fails with `untracked working tree files would be overwritten` | The notebooks now clean `data/manifests` automatically before pulling. If you still see it, run a new cell: `!cd /content/checkmaize && git clean -fdq data/manifests && git pull` and re-run the failed cell. |
| Cell 5 (smoke tests) fails with `FileNotFoundError` | The notebooks were updated to fix this. Re-run Cell 1 (git pull), then Cell 4 (rebuild the lists), then Cell 5. |
| Notebook 03 export prints `Failed to convert the model to the target version 17` or quantize crashes with `Inferred shape and existing shape differ` | torch 2.9+ defaults to the new ONNX exporter whose graphs break the quantizer's shape check. Re-run Cell 1 (git pull → `inference/export.py` now uses the legacy exporter `dynamo=False`), confirm `winner` is exactly one name, then re-run Cell 3. |
| Kaggle download requires login | Sign in to Kaggle with your Google account, then click Download on the dataset page. |
| The VS Code cell says the zip is at `/content/...` but I can't see it | Open the Colab extension's file explorer (Colab icon → the file tree view) and navigate to `/content`. |
| Dragging the dataset zip into the file explorer does nothing | The upload can take many minutes for ~1.3 GB. Watch VS Code's status bar; when the progress disappears, re-run Cell 3. |
| App crashes at startup with `Cannot read property 'install' of null` | The onnxruntime fix did not run. In the `app` folder run `node scripts/fix-onnxruntime.js`, then `npx expo run:android` again. |
| `npm test` fails on the parity test after replacing fixtures | The fixture replacement didn't match (wrong files, or you replaced with files from a different model run). Re-download `fixtures.zip` from Notebook 03 and replace again. |
| Phone won't install the APK | Android blocks unknown sources. When prompted, allow "Install unknown apps" for the browser you downloaded from. |
| Everything is slow on your computer | That's expected — all heavy work happens on the cloud runtime. Your computer only moves small files around. |

---

## Glossary (plain words)

- **Model / brain** — a program that learned to recognise patterns from thousands of examples.
- **Training** — showing the model thousands of labelled photos until it learns.
- **Epoch** — one full pass over all training photos. The score climbs with epochs.
- **Accuracy** — the share of exam photos the model got right (0.97 = 97%).
- **ONNX** — a file format for models that works everywhere (Python, phones...).
- **Quantization** — compressing the model so it runs fast on a phone with almost no accuracy loss.
- **Parity test** — a test that checks the app does the exact same photo math as Python.
- **GPU / T4** — the free fast graphics processor Colab lends you for training.
- **APK** — the installable file for Android apps (like `.exe` on Windows).
