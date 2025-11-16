import cv2
from ultralytics import YOLO

# --- [ 1 ] ---
# حمل الموديل بتاعك اللي نزلته
# (اتأكد إن الفايل في نفس الفولدر مع الكود)
print("...[+] Loading our 99.4% AI Model...")
model = YOLO('YOLOv8_Recycling_Model.pt') 
print("...[✔] Model Loaded. Starting Camera...")

# --- [ 2 ] ---
# شغل الكاميرا (رقم 0 هي الكاميرا الأساسية)
cap = cv2.VideoCapture(0) 

while True:
    # اقرا فريم (صورة) من الكاميرا
    ret, frame = cap.read()
    if not ret:
        break

    # --- [ 3 ] ---
    # ابعت الفريم للموديل عشان يتوقع (ده سطر الـ AI)
    results = model(frame)

    # --- [ 4 ] ---
    # هات النتيجة من الموديل
    top1_index = results[0].probs.top1          # هات أعلى احتمال (index)
    top1_class_name = model.names[top1_index] # هات اسم الكلاس (Metal, Plastic..)
    top1_confidence = results[0].probs.top1conf # هات درجة الثقة (e.g. 0.99)
    
    # حضر التكست اللي هيتكتب على الشاشة
    text = f"{top1_class_name} ({top1_confidence*100:.1f}%)"
    
    # --- [ 5 ] ---
    # اكتب التكست ده على الفريم
    cv2.putText(
        frame, 
        text, 
        (50, 50), # المكان (50 بيكسل من فوق و 50 من الشمال)
        cv2.FONT_HERSHEY_SIMPLEX, # نوع الخط
        1.5, # حجم الخط
        (0, 255, 0), # اللون (أخضر)
        3 # سُمك الخط
    )

    # --- [ 6 ] ---
    # اعرض الفريم اللي عليه النتيجة
    cv2.imshow('Osama AI Demo (Press Q to quit)', frame)

    # استنى دوسة 'q' عشان تخرج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# اقفل كل حاجة
cap.release()
cv2.destroyAllWindows()