# quick test to verify the WAV and playback
import os
import sys
import time
import platform

ALARM_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "alarm.wav")
ALARM_PATH = os.path.normpath(ALARM_PATH)

print("Alarm path:", ALARM_PATH)
print("Exists:", os.path.exists(ALARM_PATH))
if not os.path.exists(ALARM_PATH):
    print("ERROR: alarm.wav not found. Put a WAV file at assets/alarm.wav")
    sys.exit(1)

if platform.system() == "Windows":
    try:
        import winsound
        print("Using winsound.PlaySound...")
        winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        print("Played via winsound. Sleeping 3s...")
        time.sleep(3)
        print("Done.")
        sys.exit(0)
    except Exception as e:
        print("winsound failed:", e)

# fallback to pygame
try:
    import pygame
    # pre-init to avoid driver issues
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    s = pygame.mixer.Sound(ALARM_PATH)
    s.play()
    print("Played via pygame. Sleeping 3s...")
    time.sleep(3)
    print("Done.")
except Exception as e:
    print("pygame failed:", e)
    raise
