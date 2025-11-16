import cv2
from ultralytics import YOLO

print("...[+] Loading our 99.4% AI Model...")
model = YOLO('YOLOv8_Recycling_Model.pt') 
print("...[✔] Model Loaded. Starting Camera...")

cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    top1_index = results[0].probs.top1
    top1_class_name = model.names[top1_index]
    top1_confidence = results[0].probs.top1conf
    
    text = f"{top1_class_name} ({top1_confidence*100:.1f}%)"
    
    cv2.putText(
        frame, 
        text, 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5, 
        (0, 0, 0), 
        3 
    )

    cv2.imshow('Osama AI Demo (Press Q to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()