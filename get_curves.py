from lightkurve import search_lightcurve
import numpy as np
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
    ("Kepler-9", 1),
    ("Kepler-7", 1),
    ("Kepler-5", 1),
    ("Kepler-11", 1),
    ("Kepler-12", 1),
    ("Kepler-17", 1),
    ("Kepler-20", 1),
    ("Kepler-22", 1),
    ("Kepler-37", 1),
    ("Kepler-62", 1),
    ("Kepler-69", 1),
    ("Kepler-78", 1),
    ("Kepler-90", 1),
    ("Kepler-186", 1),
    ("Kepler-452", 1),
    ("Kepler-62f", 1),
    ("Kepler-438b", 1),
    ("Kepler-442b", 1),
    ("Kepler-62e", 1),
    ("Kepler-283c", 1),
    ("Kepler-296e", 1),
    ("Kepler-296f", 1),
    ("Kepler-440b", 1),
    ("Kepler-443b", 1),
    ("Kepler-68b", 1),
    ("Kepler-100c", 1),
    ("Kepler-407b", 1),
    ("KOI-102", 2),
    ("KOI-87", 2),
    ("KOI-94", 2),
    ("KOI-1428", 2),
    ("KOI-314", 2),
    ("KOI-7016", 2),
    ("KOI-2124", 2),
    ("KOI-268", 2),
    ("KOI-292", 2),
    ("KOI-3158", 2),
    ("KOI-217", 2),
    ("KOI-2418", 2),
    ("KOI-3512", 2),
    ("KIC 6278683", 0),
    ("KIC 9832227", 0),
    ("KIC 1026957", 0),
    ("KIC 12557548", 0),
    ("KIC 8462852", 0),
    ("KIC 6933899", 0),
    ("KIC 3241344", 0),
    ("KIC 3427720", 0),
    ("KIC 7671081", 0),
    ("KIC 12356914", 0),
    ("KIC 9705459", 0),
    ("KIC 5113061", 0),
    ("KIC 10118816", 0),
    ("KIC 2997455", 0),
]


for name, label in objects:
    try:
        filename = f"{name.replace(' ', '_')}.npz"
        if filename in os.listdir("data/curves"):
            logging.info(f"Skipping {name} (already downloaded)")
            continue

        result = search_lightcurve(name, mission="Kepler", quarter=3)
        if result is None or len(result) == 0:
            logging.info(f"No data found for {name}")
            continue

        lcfile = result.download()
        if lcfile is None:
            logging.info(f"Download failed for {name}")
            continue

        lc = lcfile.remove_nans()
        flux = lc.flux / np.nanmedian(lc.flux)
        time = lc.time.value

        np.savez(f"data/curves/{filename}", time=time, flux=flux, label=label)
        logging.info(f"Saved {name}")
    except Exception as e:
        logging.error(f"Error with {name}: {e}")

