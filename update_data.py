import requests
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIG  (set these in your .env file)
# -----------------------------
STEP       = os.getenv("FLOOD_STEP", "0.2")
BASE_URL   = os.getenv("FLOOD_API_URL", "http://:8000")
URL        = f"{BASE_URL}/florida_heatmap?step={STEP}"

OUTPUT_FILE = "flood_data.json"
BACKUP_FILE = "flood_data_backup.json"
TMP_FILE    = "flood_data.tmp.json"   # written first, then atomically renamed
LOG_FILE    = "cron_flood.log"

REQUIRED_KEYS = {"Latitude", "Longitude", "Flood_Probability"}

# -----------------------------
# LOGGING  (rotating — won't grow forever)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# -----------------------------
# FETCH & SAVE
# -----------------------------
def fetch_and_save() -> None:
    logger.info(f"Fetching heatmap data from {URL}")

    response = requests.get(URL, timeout=300)   # 5 min — appropriate for step=0.2
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"Unexpected response format or empty data: type={type(data)}, len={len(data) if isinstance(data, list) else 'n/a'}"
        )

    # Warn about any malformed points (don't abort — partial data is still useful)
    bad_indices = [i for i, pt in enumerate(data) if not REQUIRED_KEYS.issubset(pt)]
    if bad_indices:
        logger.warning(
            f"{len(bad_indices)} points are missing expected fields "
            f"(first 5 indices: {bad_indices[:5]})"
        )

    # Backup last good file before overwriting
    if os.path.exists(OUTPUT_FILE):
        shutil.copy(OUTPUT_FILE, BACKUP_FILE)
        logger.info(f"Backed up previous data to {BACKUP_FILE}")

    # Build payload
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "point_count": len(data),
        "step": STEP,
        "data": data,
    }

    # Atomic write: write to tmp first, then rename
    # Prevents corrupted output if the process crashes mid-write
    with open(TMP_FILE, "w") as f:
        json.dump(payload, f)

    os.replace(TMP_FILE, OUTPUT_FILE)  # atomic on Linux

    logger.info(f" Saved {len(data)} points to {OUTPUT_FILE}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        fetch_and_save()
    except requests.exceptions.ConnectionError:
        logger.error(" Could not connect to the API. Is the server running?")
    except requests.exceptions.Timeout:
        logger.error("Request timed out. Try increasing timeout or using a larger step size.")
    except requests.exceptions.HTTPError as e:
        logger.error(f" Server error: {e.response.status_code} — {e.response.text}")
    except ValueError as e:
        logger.error(f" Bad data received: {e}")
    except Exception as e:
        logger.exception(f" Unexpected error: {e}")
