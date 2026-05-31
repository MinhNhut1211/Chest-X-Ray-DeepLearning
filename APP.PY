from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
import cv2
import os
import matplotlib.cm as cm

# --- IMPORT THÊM CHO ViT (PYTORCH) ---
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CNN_PATH = os.path.join(BASE_DIR, 'MoHinhHuanLuyen', 'chest_xray_best_model.keras')
# Đường dẫn file trọng số ViT (.pth)
MODEL_VIT_PATH = os.path.join(BASE_DIR, 'MoHinhHuanLuyen', 'vit_chest_xray_pytorch_best.pth') 
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'results')

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Cấu hình thiết bị chạy PyTorch toàn cục
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. KHỞI TẠO MẠNG CNN TỰ XÂY DỰNG (GIỮ NGUYÊN)
# ==============================================================================
custom_cnn_model = Sequential([
    Input(shape=(150, 150, 1)),
    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

try:
    if os.path.exists(MODEL_CNN_PATH):
        custom_cnn_model.load_weights(MODEL_CNN_PATH)
        print("[+] Đã tải thành công trọng số CNN Tự xây dựng!")
    else:
        print(f"[!] Không tìm thấy file trọng số CNN tại: {MODEL_CNN_PATH}")
except Exception as e:
    print(f"Lỗi tải trọng số CNN: {e}")

# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH VISION TRANSFORMER - ViT (THAY THẾ VGG16)
# ==============================================================================
class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ViTWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x).logits

vit_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=2, 
    ignore_mismatched_sizes=True
).to(device)

try:
    if os.path.exists(MODEL_VIT_PATH):
        vit_model.load_state_dict(torch.load(MODEL_VIT_PATH, map_location=device))
        vit_model.eval() # Bắt buộc chuyển sang chế độ đánh giá (Inference)
        print("[+] Đã tải thành công trọng số ViT từ file .pth!")
    else:
        print(f"[!] Không tìm thấy file trọng số ViT tại: {MODEL_VIT_PATH}")
except Exception as e:
    print(f"[!] Lỗi tải trọng số ViT: {e}")

# ==============================================================================
# 3. CÁC HÀM XỬ LÝ GRAD-CAM CHO MẠNG CNN (GIỮ NGUYÊN)
# ==============================================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, is_vgg=False):
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output = grad_model(img_array)
        x = last_conv_layer_output
        start_index = False
        for layer in model.layers:
            if start_index:
                x = layer(x)
            if layer.name == last_conv_layer_name:
                start_index = True
        preds = x
        if pred_index is None:
            class_channel = preds[:, 0]
        else:
            class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
        
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    cv2.imwrite(cam_path, superimposed_img)
    return cam_path

# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET'])
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh!'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file!'})

    model_choice = request.form.get('model_choice', 'custom_cnn')

    try:
        temp_path = os.path.join(STATIC_DIR, 'temp_input.jpg')
        file.save(temp_path)
        
        image = cv2.imread(temp_path)
        prediction = 0
        model_name_display = ""
        heatmap_file = "gradcam_result.jpg"
        heatmap_path = os.path.join(STATIC_DIR, heatmap_file)

        # ---------------------------------------------------------
        # CHẠY MODEL 1: CNN TỰ XÂY DỰNG
        # ---------------------------------------------------------
        if model_choice == 'custom_cnn':
            model_name_display = "Mạng CNN Tự Xây Dựng"
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (150, 150))
            img_normalized = img_resized / 255.0
            img_reshaped = np.reshape(img_normalized, (1, 150, 150, 1))
            
            prediction = custom_cnn_model.predict(img_reshaped)[0][0]
            
            last_conv = "conv2d_2"
            heatmap = make_gradcam_heatmap(img_reshaped, custom_cnn_model, last_conv, is_vgg=False)
            save_and_display_gradcam(temp_path, heatmap, heatmap_path)

        # ---------------------------------------------------------
        # CHẠY MODEL 2: VISION TRANSFORMER (ViT)
        # ---------------------------------------------------------
        elif model_choice == 'vit' or model_choice == 'vgg16': 
            model_name_display = "Mô Hình Vision Transformer (ViT)"
            
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # ĐỒNG BỘ 100% CHUẨN XỬ LÝ ẢNH IMAGENET VỚI LÚC TRAIN
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Thực hiện dự đoán (Tắt tính toán gradient để tăng tốc)
            vit_model.eval()
            with torch.no_grad():
                outputs = vit_model(img_tensor).logits
                probs = F.softmax(outputs, dim=1)[0]
                
                prob_normal = probs[0].item()
                prob_pneumonia = probs[1].item()
                
                # In thông tin kiểm tra ra màn hình CMD/Terminal
                print("\n" + "="*50)
                print(f"[KIỂM TRA ViT KẾT QUẢ]")
                print(f" -> Xác suất Phổi Bình Thường: {prob_normal * 100:.2f}%")
                print(f" -> Xác suất Bị Viêm Phổi:     {prob_pneumonia * 100:.2f}%")
                print("="*50 + "\n")
                
                prediction = prob_pneumonia 
                predicted_class = torch.argmax(probs).item()

            # Tiến hành vẽ Heatmap (Nằm ngoài vùng torch.no_grad() để GradCAM tính toán)
            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            target_layers = [vit_model.vit.encoder.layer[-1].layernorm_before]
            wrapped_model = ViTWrapper(vit_model)
            cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
            
            targets = [ClassifierOutputTarget(predicted_class)]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

            # Trồng bản đồ nhiệt đè lên ảnh gốc (giữ nguyên kích thước gốc để hiển thị đẹp trên giao diện)
            grayscale_cam_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
            img_float = np.float32(img_rgb) / 255.0
            visualization = show_cam_on_image(img_float, grayscale_cam_resized, use_rgb=True)
            cv2.imwrite(heatmap_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        # ---------------------------------------------------------
        # ĐÁNH GIÁ NGƯỠNG TIN CẬY (ĐÃ TỐI ƯU NGƯỠNG PHÙ HỢP LÂM SÀNG)
        # ---------------------------------------------------------
        confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100
        
        # Áp dụng ngưỡng 0.50 chuẩn nhị phân (Binary Classification) để tối đa hoá độ chính xác
        if prediction >= 0.50:
            result_class = 'CÓ DẤU HIỆU VIÊM PHỔI (Pneumonia)'
            status = 'danger'
        else:
            result_class = 'PHỔI BÌNH THƯỜNG (Normal)'
            status = 'success'
            
        return jsonify({
            'result': result_class,
            'status': status,
            'confidence': f"{confidence:.2f}%",
            'heatmap_url': f'/static/results/{heatmap_file}?t={np.random.randint(10000)}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc() # In chi tiết lỗi hệ thống ra console để dễ xử lý
        return jsonify({'error': f"Đã xảy ra lỗi hệ thống: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)