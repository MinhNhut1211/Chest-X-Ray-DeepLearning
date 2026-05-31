# 🫁 Pneumonia X-Ray AI Diagnostic (Python Flask)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

Ứng dụng hỗ trợ chẩn đoán viêm phổi qua ảnh chụp X-quang ngực được xây dựng trên nền tảng Web Flask. Công cụ này giúp y bác sĩ và nhà nghiên cứu dễ dàng phân loại tình trạng bệnh lý, đồng thời đối chiếu độ chính xác giữa mô hình **Mạng nơ-ron tích chập (CNN)** và **Vision Transformer (ViT)**.

---

## 🚀 Tính năng nổi bật

- **Dual Model Inference:** Hỗ trợ dự đoán đồng thời bằng hai kiến trúc Deep Learning tiên tiến nhất: Custom CNN (TensorFlow/Keras) và Vision Transformer (PyTorch).
- **Explainable AI (Grad-CAM):** Tự động tính toán gradient và vẽ bản đồ nhiệt (heatmap) định vị chính xác vùng phổi bị tổn thương, xóa bỏ tính "hộp đen" của AI để tăng độ tin cậy trong y tế.
- **Clinical Threshold Evaluation:** Đánh giá ngưỡng lâm sàng tự động, phân loại kết quả thành: 
  - 🔴 `CÓ DẤU HIỆU VIÊM PHỔI` (Xác suất >= 0.50)
  - 🟢 `PHỔI BÌNH THƯỜNG` (Xác suất < 0.35)
  - 🟡 `KẾT QUẢ KHÔNG RÕ RÀNG` (Nằm trong khoảng xám)
- **Friendly Web Interface:** Trải nghiệm người dùng mượt mà theo mô hình Client-Server, cho phép upload ảnh X-quang trực tiếp qua trình duyệt web và nhận kết quả tức thì.

---

## 🏗️ Cấu trúc Mã nguồn (Source Code Architecture)

### 1. AI Training & Models (`doan.ipynb`)
Sổ tay Jupyter chứa toàn bộ quá trình tiền xử lý dữ liệu và huấn luyện mô hình trên Google Colab.
- **Data Preprocessing:** Chuẩn hóa ảnh (resize 150x150 cho CNN, 224x224 cho ViT), chuẩn hóa pixel [0, 1] và lật ảnh ngẫu nhiên để tăng cường dữ liệu.
- **Model Architecture:** Xây dựng chuỗi tích chập tuần tự cho CNN và fine-tuning mô hình `vit-base-patch16-224` của HuggingFace.

### 2. Core Backend Logic (`APP.py`)
- **Model Loader:** Tải trọng số tốt nhất đã được huấn luyện (`chest_xray_best_model.keras` và `vit_chest_xray_pytorch_best.pth`) vào bộ nhớ.
- **Inference & Heatmap Generator:** Tiếp nhận ảnh từ Frontend, gọi mô hình suy luận (Inference), sau đó sử dụng thư viện `pytorch_grad_cam` để xuất bản đồ nhiệt và lưu vào thư mục `/static/results/`.

### 3. Giao diện người dùng (Frontend)
- **HTML/CSS/JS:** Được render qua Flask, cung cấp vùng thả ảnh (Drag & Drop) thông minh và vùng hiển thị kết quả phân loại đi kèm độ tin cậy (Confidence %).

---

## 🛠️ Hướng dẫn cài đặt & Sử dụng

**1. Yêu cầu hệ thống:** Máy tính cài đặt Python 3.8+ và các thư viện cần thiết.

**2. Cài đặt thư viện:** 
Mở Terminal/CMD và chạy lệnh sau:
pip install Flask numpy tensorflow torch torchvision transformers pytorch-grad-cam opencv-python matplotlib Pillow

**3. 📷 Hình ảnh Demo:**
<img width="948" height="423" alt="image" src="https://github.com/user-attachments/assets/6e813d80-5cb2-4dce-a950-76568c8085a2" />

**4. 🚀 File Dataset:** 
https://drive.google.com/file/d/1jY__cBgpJMJzPfeXhisZXsxYUUZsMnPD/view?usp=drive_link


# Contact <p>
##### Facebook: https://www.facebook.com/tranminhnhut121102 <p>
##### Stk: 0601000528876
##### Nh: Vietcombank
##### Ten: Tran Minh Nhut
