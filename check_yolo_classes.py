from ultralytics import YOLO
model = YOLO('/home/ubuntu/videoforprocessing_link/football_analysis_v2/data/models/best_player_detect.pt')
print(f"Model classes: {model.names}")
