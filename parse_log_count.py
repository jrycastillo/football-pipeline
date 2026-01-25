import re

LOG_FILE = "polling_service.log"
TARGET_USER = "01c073e5"

def parse_log():
    last_count = None
    try:
        with open(LOG_FILE, "r", errors="ignore") as f:
            for line in f:
                if "Filtered" in line and TARGET_USER in line:
                    # Pattern: [poll] Found X total. Filtered Y for user 01c073e5.
                    match = re.search(r"Filtered (\d+) for user", line)
                    if match:
                        last_count = int(match.group(1))
    except FileNotFoundError:
        print("Log file not found.")
        return

    if last_count is not None:
        print(f"User {TARGET_USER} has {last_count} videos (from latest poll).")
    else:
        print(f"No 'Filtered' messages found for user {TARGET_USER}.")

if __name__ == "__main__":
    parse_log()
