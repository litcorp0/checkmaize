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
- A computer with internet and Chrome/Firefox
- A free Google account (for Colab)
- A free GitHub account (so Colab can read your project code)
- An Android phone (Android 7 or newer) with a USB cable
- A free Kaggle account (Google login — to download the Ghana dataset)

---

## Part 0 — Put the project on GitHub (one-time, ~15 minutes)

Colab cannot see the files on your computer. It can, however, download (clone)
a project from GitHub. So we first upload the project to GitHub.

1. Go to <https://github.com> and sign up / sign in.
2. Click the green **New** button (top left), or the **+** button in the top
   right corner → **New repository**.
3. Repository name: `checkmaize`. Choose **Private** (keeps your work hidden).
   Do **not** tick "Add a README" or any other box. Click **Create repository**.
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

> **No terminal on your computer?** Skip Part 0. In every Colab notebook below,
> use the "upload a zip" option shown in Cell 1 instead. Zip the whole
> `checkmaize` folder (right-click → Compress), upload it in the Colab files
> pane, and follow the OPTION B instructions printed by the notebook.

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
4. Save the file `crop-pest-and-disease-detection.zip` (~1.3 GB) in your
   **Downloads** folder. **Remember where it went.**

> **Fallback route — Mendeley (only if Kaggle is unavailable):** go to
> <https://data.mendeley.com/datasets/bwh3zbpkpv/1>, sign in (Google works),
> and use "Download All Files" (about 8 GB — it includes an extra augmented
> copy we don't need, so prefer Kaggle). If you use this route, rename the
> downloaded file to `Raw Data.zip`.

### Step 1.2 — Open the notebook in Colab

1. Go to <https://colab.research.google.com> and sign in with your Google account.
2. Menu: **File → Upload notebook**.
3. Click **Browse**, find the file `colab/01_dataset.ipynb` inside your project
   folder, and open it.
4. The notebook opens as a list of grey boxes ("cells"). You run them one by
   one with the **play button (▶)** on the left of each cell. **Wait for each
   cell to finish before running the next one.**

### Step 1.3 — Run Cell 1 (connect to your code)

Press ▶ on Cell 1.

- If you pushed the repo to GitHub (Part 0), this cell prints instructions.
  Copy the `git clone` line it shows, paste it into a **new cell** (button
  **+ Code**), and run it. Then run the `pip install` line it shows.
- If you did not push to GitHub: in the left sidebar click the **folder icon**,
  then the **upload icon (⤒)**, upload `checkmaize.zip`, and run the OPTION B
  commands the cell prints.

Expected result: no red error text. (Red = problem; see Troubleshooting.)

### Step 1.4 — Run Cell 2 (download PlantVillage, automatic)

Press ▶ on Cell 2. This downloads ~2 GB of images. **Expect 10–30 minutes.**

Expected result, at the bottom of the cell:

```
train: 4000 corn rows (approximately)
test: 1000 corn rows (approximately)
plantvillage extraction done
```

### Step 1.5 — Upload the Ghana dataset and run Cell 3

1. In the left sidebar, click the **folder icon**.
2. Use the **upload (⤒)** button to upload the dataset zip from your computer
   (`crop-pest-and-disease-detection.zip` from Kaggle, or `Raw Data.zip` from
   Mendeley). (Large file — be patient, the progress bar appears bottom-left
   of Colab.)
3. Press ▶ on Cell 3. The notebook accepts both layouts automatically and
   keeps only the three classes we need.

Expected result — three lines (numbers should match exactly):

```
Using: crop-pest-and-disease-detection.zip
leaf blight 1006
leaf spot 1259
healthy 208
ccmt_ghana ready: {'Leaf blight': 1006, 'Leaf spot': 1259, 'Healthy': 208}
```

(If you see `No dataset zip found in /content` — the upload was not finished.
Wait for the progress bar to complete, then re-run the cell.)

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

### Step 1.7 — Run Cell 5 (download the result)

Press ▶ on Cell 5. It zips the six list files and downloads `splits.zip` to
your computer's Downloads folder.

### Step 1.8 — Put the lists into your project

1. Unzip `splits.zip` (right-click → Extract All).
2. Inside it is a folder `manifests` with files like `train.csv`, `test.csv`.
   Copy **all of them** into your project folder: `checkmaize/data/manifests/`.
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

### Step 2.1 — Turn on the free GPU

1. Open `colab/02_train_benchmark.ipynb` in Colab (Upload notebook, same as
   before).
2. **Important:** Menu: **Runtime → Change runtime type → Hardware accelerator
   → T4 GPU → Save.** Without the GPU, training takes days instead of hours.

### Step 2.2 — Run Cell 1 (connect + check the GPU)

Run Cell 1 (same options as Notebook 01). Then in a new cell run:

```python
import torch
print(torch.cuda.is_available())
```

Expected: `True`. If it prints `False`, you forgot Step 2.1.

### Step 2.3 — Run Cell 2 (quick self-test of the code)

Run Cell 2. Expected: all tests `passed`. (This re-downloads nothing big; it
reuses the committed lists. It also re-extracts the picture files — Cell 1 of
Notebook 01's downloads do **not** carry over between notebooks; Colab forgets
everything when a notebook closes. If Cell 2 complains it cannot find
`data/raw/...`, re-run Notebook 01's Cells 2 and 3 first.)

### Step 2.4 — Run Cell 3 (the big training run)

Run Cell 3. **This is the long one — 2 to 3 hours.** Colab may disconnect if
you leave the tab; keep the tab open in the foreground if you can. If it stops,
just re-run the cell — finished models are saved and skipped... (no — each
model trains from scratch; if it stops mid-model, re-run the cell and it starts
that model over).

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

### Step 2.6 — Run Cell 4 (download the scoreboard + the trained brains)

Run Cell 4. It downloads `report.zip` and `runs.zip` to your computer.

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

Upload `colab/03_export.ipynb` to Colab and open it.

### Step 3.2 — Run Cell 1 (connect + install the small tools)

Same as before — connect to the repo (or upload the zip), install.

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

### Step 3.6 — Run Cell 5 (download everything)

Run Cell 5. It downloads:
- `artifacts.zip` — contains `model_int8.onnx` (the squeezed brain), `labels.json`, `metrics.json`
- `fixtures.zip` — the parity-test pictures the app uses to double-check its math
- `onnx-contract.md` — the technical agreement between Python and the app

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
| Notebook 02 Cell 2 cannot find `data/raw/...` | Colab forgets downloads between notebooks. Re-run Notebook 01 Cells 2 and 3, then continue. |
| `torch.cuda.is_available()` prints `False` | You forgot Runtime → Change runtime type → T4 GPU. Change it and re-run from Cell 1. |
| Colab disconnects mid-training | Re-open the notebook, re-run Cell 1, then re-run Cell 3 — it restarts the interrupted model. |
| Kaggle download requires login | Sign in to Kaggle with your Google account, then click Download on the dataset page. |
| `files.download` doesn't start | Allow pop-ups/downloads for colab.research.google.com in the browser. |
| App crashes at startup with `Cannot read property 'install' of null` | The onnxruntime fix did not run. In the `app` folder run `node scripts/fix-onnxruntime.js`, then `npx expo run:android` again. |
| `npm test` fails on the parity test after replacing fixtures | The fixture replacement didn't match (wrong files, or you replaced with files from a different model run). Re-download `fixtures.zip` from Notebook 03 and replace again. |
| Phone won't install the APK | Android blocks unknown sources. When prompted, allow "Install unknown apps" for the browser you downloaded from. |
| Everything is slow on your computer | That's expected — all heavy work happens in Colab. Your computer only moves small files around. |

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
