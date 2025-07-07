import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import os
import json
import time
from datetime import datetime
from PIL import Image
import requests
import asyncio
import websockets

# Tripo API Configuration
API_KEY = "tsk_xuC9uihGXe3Nk49oejYKjdrAz2p29pgjhnv-y3bP2Z1"

class FurnitureTo3DPipeline:
    def __init__(self, 
                 sam_checkpoint="sam_vit_b_01ec64.pth",
                 model_type="vit_b",
                 yolo_model_path="best.pt",
                 output_dir="furniture_3d_pipeline"):
        
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.yolo_model_path = yolo_model_path
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        self.png_dir = os.path.join(output_dir, "extracted_furniture")
        self.model_3d_dir = os.path.join(output_dir, "3d_models")
        self.log_file = os.path.join(output_dir, "processing_log.json")
        
        os.makedirs(self.png_dir, exist_ok=True)
        os.makedirs(self.model_3d_dir, exist_ok=True)
        
        # Initialize processing log
        self.processing_log = {
            "session_start": datetime.now().isoformat(),
            "processed_items": []
        }
        
        print(f"🔧 Using device: {self.device}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def shrink_box(self, box, factor=1.0):
        """Box küçültme fonksiyonu"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * factor, (y2 - y1) * factor
        new_x1 = int(cx - w / 2)
        new_y1 = int(cy - h / 2)
        new_x2 = int(cx + w / 2)
        new_y2 = int(cy + h / 2)
        return [new_x1, new_y1, new_x2, new_y2]
    
    def extract_furniture_png(self, image, mask, box, class_name, index):
        """Segmente edilmiş mobilyayı PNG olarak kaydet"""
        x1, y1, x2, y2 = box
        
        # Bounding box boyutlarını al
        width = x2 - x1
        height = y2 - y1
        
        # Minimum boyut kontrolü (çok küçük objeler için)
        if width < 50 or height < 50:
            print(f"⚠️ Object too small ({width}x{height}), skipping: {class_name}_{index}")
            return None
        
        # Bounding box bölgesini crop et
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        
        # RGBA formatında yeni görüntü oluştur
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[:, :, :3] = cropped_image
        rgba_image[:, :, 3] = cropped_mask * 255
        
        # Dosya adını oluştur
        filename = f"{class_name}_{index:03d}.png"
        filepath = os.path.join(self.png_dir, filename)
        
        # PIL Image ile kaydet
        pil_image = Image.fromarray(rgba_image, 'RGBA')
        pil_image.save(filepath, 'PNG')
        
        print(f"✅ PNG saved: {filename} ({width}x{height})")
        return filepath
    
    def segment_furniture(self, image_path):
        """Mobilyaları segmente et ve PNG'leri oluştur"""
        print("=" * 60)
        print("🔍 FURNITURE SEGMENTATION PHASE")
        print("=" * 60)
        
        try:
            # Dosya kontrolü
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # SAM Model yükle
            print("📥 Loading SAM model...")
            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            sam.to(self.device)
            predictor = SamPredictor(sam)
            
            # Görüntü yükle
            print("🖼️ Loading image...")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)
            
            # YOLO model yükle
            print("🎯 Loading YOLO model...")
            yolo_model = YOLO(self.yolo_model_path)
            results = yolo_model(image_path)[0]
            
            # Model sınıflarını göster
            print(f"📊 Model classes: {list(yolo_model.names.values())}")
            print(f"🔢 Total classes: {len(yolo_model.names)}")
            
            if len(results.boxes) == 0:
                print("❌ No objects detected!")
                return []
            
            print(f"🎯 Found {len(results.boxes)} objects")
            
            extracted_files = []
            
            # Her obje için işlem yap
            for i, box_tensor in enumerate(results.boxes.xyxy):
                try:
                    # Box koordinatlarını al
                    box = box_tensor.cpu().numpy().astype(int)
                    
                    # Sınıf bilgisi
                    class_id = int(results.boxes.cls[i])
                    class_name = yolo_model.names[class_id] if class_id in yolo_model.names else "unknown"
                    confidence = float(results.boxes.conf[i]) if hasattr(results.boxes, 'conf') else 0.0
                    
                    # Düşük güven skorlu objeleri atla
                    if confidence < 0.5:
                        print(f"⚠️ Low confidence ({confidence:.2f}) for {class_name}, skipping...")
                        continue
                    
                    # Box'ı ayarla
                    box = self.shrink_box(box, factor=1.0)
                    x1, y1, x2, y2 = box
                    
                    # Koordinatları sınırlar içinde tut
                    h, w = image_rgb.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Geçerli box kontrolü
                    if x2 <= x1 or y2 <= y1:
                        print(f"⚠️ Invalid box for object {i}: ({x1}, {y1}, {x2}, {y2})")
                        continue
                    
                    input_box = np.array([x1, y1, x2, y2])
                    
                    # SAM ile segmentasyon
                    print(f"🔍 Processing {class_name} #{i+1} (confidence: {confidence:.2f})...")
                    masks, scores, _ = predictor.predict(
                        box=input_box,
                        multimask_output=False
                    )
                    
                    # En iyi maskı seç
                    selected_mask = None
                    
                    for mask in masks:
                        if mask.shape != image_rgb.shape[:2]:
                            continue
                        
                        cropped = mask[y1:y2, x1:x2]
                        score = cropped.sum() / ((y2 - y1) * (x2 - x1) + 1e-6)
                        
                        if 0.05 < score < 0.5:
                            selected_mask = mask
                            break
                    
                    if selected_mask is None and len(masks) > 0:
                        selected_mask = masks[0]
                    
                    if selected_mask is None:
                        print(f"❌ No valid mask found for {class_name} #{i+1}")
                        continue
                    
                    # PNG olarak kaydet
                    filepath = self.extract_furniture_png(
                        image_rgb, 
                        selected_mask.astype(np.uint8), 
                        [x1, y1, x2, y2], 
                        class_name, 
                        i + 1
                    )
                    
                    if filepath:
                        extracted_files.append({
                            'filepath': filepath,
                            'filename': os.path.basename(filepath),
                            'class_name': class_name,
                            'box': [x1, y1, x2, y2],
                            'index': i + 1,
                            'confidence': confidence
                        })
                    
                except Exception as e:
                    print(f"❌ Error processing object {i}: {str(e)}")
                    continue
            
            print(f"\n🎉 Successfully extracted {len(extracted_files)} furniture pieces!")
            
            # Save bounding box coordinates to JSON file for interactive viewer
            coordinates_file = os.path.join(self.output_dir, "furniture_coordinates.json")
            coordinates_data = {
                "image_path": image_path,
                "image_dimensions": {
                    "width": image_rgb.shape[1],
                    "height": image_rgb.shape[0]
                },
                "furniture_pieces": []
            }
            
            for item in extracted_files:
                furniture_info = {
                    "filename": item['filename'],
                    "class_name": item['class_name'],
                    "bounding_box": {
                        "x1": int(item['box'][0]),
                        "y1": int(item['box'][1]),
                        "x2": int(item['box'][2]),
                        "y2": int(item['box'][3])
                    },
                    "confidence": float(item['confidence']),
                    "center": {
                        "x": int((item['box'][0] + item['box'][2]) / 2),
                        "y": int((item['box'][1] + item['box'][3]) / 2)
                    }
                }
                coordinates_data["furniture_pieces"].append(furniture_info)
            
            # Save coordinates data
            with open(coordinates_file, 'w') as f:
                json.dump(coordinates_data, f, indent=2)
            print(f"📍 Furniture coordinates saved to: {coordinates_file}")
            
            return extracted_files
            
        except Exception as e:
            print(f"❌ Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Tripo API Functions
    def upload_image(self, image_path):
        """Görseli Tripo API'ye yükle"""
        url = "https://api.tripo3d.ai/v2/openapi/upload/sts"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        try:
            with open(image_path, "rb") as f:
                files = {"file": f}
                response = requests.post(url, headers=headers, files=files)
                
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    file_token = result["data"].get("token") or result["data"].get("image_token")
                    print(f"✅ Image uploaded. file_token: {file_token}")
                    return file_token
            
            print(f"❌ Image upload failed: {response.text}")
            return None
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None
    
    def create_image_to_model_task(self, file_token):
        """3D model üretim task'ı başlat"""
        url = "https://api.tripo3d.ai/v2/openapi/task"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        data = {
            "type": "image_to_model",
            "file": {
                "type": "png",
                "file_token": file_token
            },
            "model_version": "v2.0-20240919",
            "texture": True,
            "pbr": True,
            "texture_quality": "detailed"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200 and response.json().get("code") == 0:
                task_id = response.json()["data"]["task_id"]
                print(f"🧠 Model generation task_id: {task_id}")
                return task_id
            
            print(f"❌ Model generation request failed: {response.text}")
            return None
            
        except Exception as e:
            print(f"❌ Task creation error: {e}")
            return None
    
    async def watch_task(self, task_id, timeout=600):
        """Task ilerleyişini WebSocket üzerinden izle"""
        url = f"wss://api.tripo3d.ai/v2/openapi/task/watch/{task_id}"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        print("📡 Watching task progress...")
        
        try:
            async with websockets.connect(url, extra_headers=headers) as ws:
                try:
                    while True:
                        message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        data = json.loads(message)
                        status = data["data"]["status"]
                        print(f"🔄 Status: {status}")
                        
                        if status not in ["queued", "running"]:
                            return data["data"]
                            
                except asyncio.TimeoutError:
                    print("⏱️ Timeout: WebSocket didn't respond.")
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"🔌 Connection closed: {e}")
                except Exception as e:
                    print(f"⚠️ WebSocket error: {e}")
                    
        except Exception as e:
            print(f"❌ Couldn't connect to WebSocket: {e}")
        
        return None
    
    def download_3d_model(self, model_url, filename):
        """3D modeli indir"""
        try:
            response = requests.get(model_url)
            if response.status_code == 200:
                filepath = os.path.join(self.model_3d_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"📥 3D model downloaded: {filename}")
                return filepath
            else:
                print(f"❌ Download failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    async def process_single_furniture(self, furniture_info):
        """Tek bir mobilyayı 3D'ye çevir"""
        filename = furniture_info['filename']
        filepath = furniture_info['filepath']
        class_name = furniture_info['class_name']
        
        print(f"\n🚀 Processing {filename} ({class_name})...")
        
        # 1. Upload image
        file_token = self.upload_image(filepath)
        if not file_token:
            return None
        
        # 2. Create task
        task_id = self.create_image_to_model_task(file_token)
        if not task_id:
            return None
        
        # 3. Watch task
        result = await self.watch_task(task_id)
        if not result:
            return None
        
        # 4. Download models
        output = result.get("output", {})
        downloaded_models = {}
        
        base_filename = filename.replace('.png', '')
        
        # Download base model
        if output.get("model"):
            model_filename = f"{base_filename}_base.glb"
            model_path = self.download_3d_model(output["model"], model_filename)
            if model_path:
                downloaded_models["base"] = model_path
        
        # Download PBR model
        if output.get("pbr_model"):
            pbr_filename = f"{base_filename}_pbr.glb"
            pbr_path = self.download_3d_model(output["pbr_model"], pbr_filename)
            if pbr_path:
                downloaded_models["pbr"] = pbr_path
        
        # Log results
        processing_result = {
            "filename": filename,
            "class_name": class_name,
            "task_id": task_id,
            "processed_at": datetime.now().isoformat(),
            "downloaded_models": downloaded_models,
            "tripo_output": output
        }
        
        self.processing_log["processed_items"].append(processing_result)
        
        return processing_result
    
    def save_log(self):
        """İşlem logunu kaydet"""
        with open(self.log_file, 'w') as f:
            json.dump(self.processing_log, f, indent=2)
        print(f"📋 Processing log saved: {self.log_file}")
    
    async def process_all_furniture(self, image_path):
        """Tam pipeline - segmentasyon + 3D dönüşüm"""
        print("🎯 Starting Complete Furniture to 3D Pipeline")
        print("=" * 60)
        
        # 1. Segment furniture
        extracted_files = self.segment_furniture(image_path)
        
        if not extracted_files:
            print("❌ No furniture extracted, stopping pipeline.")
            return
        
        print("\n" + "=" * 60)
        print("🔥 3D MODEL GENERATION PHASE")
        print("=" * 60)
        
        # 2. Process each furniture piece
        total_files = len(extracted_files)
        successful_conversions = 0
        
        for i, furniture_info in enumerate(extracted_files):
            print(f"\n📊 Processing {i+1}/{total_files}")
            
            try:
                result = await self.process_single_furniture(furniture_info)
                if result:
                    successful_conversions += 1
                    print(f"✅ Successfully processed: {furniture_info['filename']}")
                else:
                    print(f"❌ Failed to process: {furniture_info['filename']}")
                
                # Rate limiting - wait between requests
                if i < total_files - 1:
                    print("⏳ Waiting 10 seconds before next request...")
                    await asyncio.sleep(10)
                    
            except Exception as e:
                print(f"❌ Error processing {furniture_info['filename']}: {e}")
                continue
        
        # 3. Save log and show summary
        self.save_log()
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED!")
        print("=" * 60)
        print(f"📊 Total furniture pieces: {total_files}")
        print(f"✅ Successfully converted: {successful_conversions}")
        print(f"❌ Failed conversions: {total_files - successful_conversions}")
        print(f"📁 PNG files: {self.png_dir}")
        print(f"📁 3D models: {self.model_3d_dir}")
        print(f"📋 Process log: {self.log_file}")

# Ana kullanım
async def main():
    # Pipeline'ı başlat
    pipeline = FurnitureTo3DPipeline(
        sam_checkpoint="sam_vit_b_01ec64.pth",
        model_type="vit_b",
        yolo_model_path="best.pt",
        output_dir="furniture_3d_pipeline"
    )
    
    # Tam pipeline'ı çalıştır
    await pipeline.process_all_furniture("image.png")

if __name__ == "__main__":
    asyncio.run(main())