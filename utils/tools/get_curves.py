from lightkurve import search_lightcurve
from tqdm import tqdm
import numpy as np
import random
import logging
import os

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d - %(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="development.log",
    level=logging.DEBUG
)

os.makedirs("data/curves", exist_ok=True)

objects = [
    ("Kepler-9", 1), ("Kepler-7", 1), ("Kepler-5", 1), ("Kepler-11", 1), ("Kepler-12", 1),
    ("Kepler-17", 1), ("Kepler-20", 1), ("Kepler-22", 1), ("Kepler-37", 1), ("Kepler-62", 1),
    ("Kepler-69", 1), ("Kepler-78", 1), ("Kepler-90", 1), ("Kepler-186", 1), ("Kepler-452", 1),
    ("Kepler-62f", 1), ("Kepler-438b", 1), ("Kepler-442b", 1), ("Kepler-62e", 1),
    ("Kepler-283c", 1), ("Kepler-296e", 1), ("Kepler-296f", 1), ("Kepler-440b", 1),
    ("Kepler-443b", 1), ("Kepler-68b", 1), ("Kepler-100c", 1), ("Kepler-407b", 1),

    ("KOI-102", 2), ("KOI-87", 2), ("KOI-94", 2), ("KOI-1428", 2), ("KOI-314", 2),
    ("KOI-7016", 2), ("KOI-2124", 2), ("KOI-268", 2), ("KOI-292", 2), ("KOI-3158", 2),
    ("KOI-217", 2), ("KOI-2418", 2), ("KOI-3512", 2),

    ("KIC 6278683", 0), ("KIC 9832227", 0), ("KIC 1026957", 0), ("KIC 12557548", 0),
    ("KIC 8462852", 0), ("KIC 6933899", 0), ("KIC 3241344", 0), ("KIC 3427720", 0),
    ("KIC 7671081", 0), ("KIC 12356914", 0), ("KIC 9705459", 0),
    ("KIC 5113061", 0), ("KIC 10118816", 0), ("KIC 2997455", 0),
]

missions = ["Kepler", "K2"]
max_per_object = 3
random.shuffle(objects)

def download_lightcurve(name, label):
    for mission in missions:
        try:
            results = search_lightcurve(name, mission=mission)
            if results is None or len(results) == 0:
                logging.info(f"No data for {name} in {mission}")
                continue

            # Limita a un número razonable de curvas
            for i, result in enumerate(results[:max_per_object]):
                filename = f"{name.replace(' ', '_')}_{mission}_{i}.npz"
                filepath = os.path.join("data/curves", filename)
                if os.path.exists(filepath):
                    continue

                lcfile = result.download()
                if lcfile is None:
                    continue

                lc = lcfile.remove_nans()
                flux = lc.flux / np.nanmedian(lc.flux)
                time = lc.time.value

                np.savez(filepath, time=time, flux=flux, label=label)
                logging.info(f"Saved {filename}")
        except Exception as e:
            logging.error(f"Error with {name} ({mission}): {e}")

# === DESCARGA ===
print("Starting bulk lightcurve download...")
for name, label in tqdm(objects):
    download_lightcurve(name, label)

print("All downloads completed.")
