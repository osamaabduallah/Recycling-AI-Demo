import gradio as gr
from ultralytics import YOLO

print("...[+] Loading our 99.4% AI Model...")
model = YOLO('YOLOv8_Recycling_Model.pt') 
print("...[✔] Model Loaded. Building Web UI...")

def classify_image(image_from_upload):
    
    results = model(image_from_upload)
    
    top1_index = results[0].probs.top1
    top1_class_name = model.names[top1_index]
    top1_confidence = results[0].probs.top1conf
    
    print(f"...[+] Prediction: {top1_class_name} ({top1_confidence*100:.1f}%)")
    return {top1_class_name: float(top1_confidence)}


iface = gr.Interface(
    fn=classify_image,                   
    inputs=gr.Image(label="Upload Image"), 
    outputs=gr.Label(num_top_classes=1, label="Result"), 
    
    title="Osama's AI Recycling Sorter ♻️",
    description="This is the (MVP/Demo) for the AI model, trained to 99.4% accuracy.",
    examples=[]
)

print("...[🚀] Starting Gradio Web App... Go to the URL below in your browser.")
iface.launch()