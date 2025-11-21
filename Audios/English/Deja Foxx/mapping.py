import os

# --- EDIT if your folder name is different ---
folder = "clips"   # folder that contains Untitled*.wav and labels.txt
labels_file = os.path.join(folder, "Labels 1.txt")  # change if your labels file has another name

# --- read timestamps ---
with open(labels_file, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

timestamps = []
for ln in lines:
    parts = ln.split()
    if len(parts) >= 2:
        start = parts[0]
        end = parts[1]
        timestamps.append((start, end))
    else:
        # if line malformed, keep as is
        timestamps.append((parts[0], ""))

# --- read wav files and sort by name (natural order) ---
wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
import re
def extract_number(filename):
    # extract the first number in the filename, e.g., "Untitled12.wav" -> 12
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

wav_files.sort(key=extract_number)
# --- sanity check ---
if len(wav_files) != len(timestamps):
    print("WARNING: number of wav files ({}) != number of timestamp lines ({})".format(len(wav_files), len(timestamps)))
    print("Proceeding by pairing by order for the minimum length.")
pairs = min(len(wav_files), len(timestamps))

# --- write mapping ---
outpath = os.path.join(folder, "mapping.txt")
with open(outpath, "w", encoding="utf-8") as out:
    out.write("file\tstart\tend\n")
    for i in range(pairs):
        out.write(f"{wav_files[i]}\t{timestamps[i][0]}\t{timestamps[i][1]}\n")

print(f"Done. Created mapping file: {outpath}")
print(f"{pairs} entries written. ({len(wav_files)} wav files, {len(timestamps)} timestamp lines)")
