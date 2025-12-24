import sys
sys.path.append("/home/ubuntu/football")
print("Importing ThreadedVideoReader...")
try:
    from utils.threaded_reader import ThreadedVideoReader
    print("Success.")
except ImportError as e:
    print(f"Failed: {e}")

print("Testing Video Open...")
v_path = "/home/ubuntu/videoforprocessing_link/U20 AFCON 2025 Qualifiers - UNAF - TUNISIA VS ALGERIA [MIzIja6ms8M].mp4"
try:
    reader = ThreadedVideoReader(v_path)
    import time
    time.sleep(2)
    print(f"Queue Size: {reader.queue.qsize()}")
    frame = reader.read()
    print(f"Frame Shape: {frame.shape if frame is not None else 'None'}")
    reader.stop()
except Exception as e:
    print(f"Runtime Error: {e}")
