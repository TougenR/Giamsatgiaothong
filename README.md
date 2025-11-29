<img width="620" height="772" alt="image" src="https://github.com/user-attachments/assets/13d277e4-d8b8-4909-9980-4dc7b265119d" />



https://github.com/user-attachments/assets/1300e33a-166f-4aeb-a099-485b90338593




# Hệ thống Giám sát Giao thông Thông minh

## Giới thiệu

Dự án **Giám sát Giao thông** là một hệ thống phát hiện phương tiện giao thông và nhận diện biển số xe sử dụng công nghệ AI. Hệ thống có khả năng xử lý video và hình ảnh để tự động xác định các loại phương tiện (ô tô, xe tải, xe buýt, xe máy) và trích xuất thông tin biển số.

## Tính năng chính

- 🔍 **Phát hiện phương tiện**: Nhận diện 4 loại phương tiện chính (ô tô, xe tải, xe buýt, xe máy)
- 📋 **Nhận diện biển số**: Phát hiện chính xác vị trí biển số xe
- 🔤 **OCR biển số**: Trích xuất văn bản từ biển số đã phát hiện
- 📊 **Thống kê**: Đếm và hiển thị số lượng phương tiện theo loại
- 🎥 **Xử lý video**: Hỗ trợ xử lý video theo thời gian thực hoặc hàng loạt
- 💻 **Tối ưu GPU**: Tự động phát hiện và sử dụng GPU để tăng tốc độ xử lý

## Kiến trúc hệ thống

```
src/
├── main.py                          # Điểm nhập chính
├── detector/                        # Module phát hiện
│   ├── vehicle_detector.py         # Phát hiện phương tiện
│   └── license_plate_detector.py   # Phát hiện biển số
├── drawer/                          # Module hiển thị
│   ├── vehicle_drawer.py           # Vẽ box phương tiện
│   └── license_plate_drawer.py     # Vẽ box biển số
├── License_plate_ocr/              # Module OCR
│   └── license_plate_ocr.py        # Trích xuất văn bản biển số
└── utils/                          # Tiện ích
    ├── gpu_utils.py                # Tối ưu GPU
    └── stub_utils.py               # Cache utilities
```

## Công nghệ sử dụng

### Core Technologies
- **Python 3.x**: Ngôn ngữ lập trình chính
- **PyTorch**: Framework deep learning
- **OpenCV**: Xử lý hình ảnh và video
- **NumPy**: Xử lý ma trận và tính toán

### AI/ML Models
- **YOLO (You Only Look Once)**:
  - Mô hình phát hiện phương tiện: `models/vehicle_model.pt`
  - Mô hình phát hiện biển số: `models/License_plate_model.pt`
- **EasyOCR**: Nhận diện ký tự trên biển số

### Performance Optimization
- **GPU Acceleration**: Tự động phát hiện và sử dụng NVIDIA CUDA
- **Memory Optimization**: Tối ưu bộ nhớ cho inference
- **Batch Processing**: Xử lý hàng loạt hiệu quả

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA (tùy chọn, cho GPU acceleration)
- Bộ nhớ RAM tối thiểu 4GB

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Sử dụng

### Chạy hệ thống
```bash
cd src
python main.py
```

Hệ thống cung cấp các tùy chọn:
1. Xử lý 100 frame đầu của video test (nhanh)
2. Xử lý toàn bộ video test (đầy đủ)
3. Xử lý video/hình ảnh tùy chỉnh
4. Thoát

### Cấu trúc đầu ra
- Video kết quả được lưu trong `output_predict_video/`
- Tên file theo định dạng `pipeline_output_[số thứ tự].mp4`

## Ứng dụng thực tế

### Giao thông thông minh
- 🚦 **Điều khiển giao thông tự động**: Thu thập dữ liệu luồng phương tiện
- 📊 **Thống kê giao thông**: Phân tích mật độ và loại phương tiện
- 🚗 **Quản lý bãi đỗ xe**: Tự động nhận diện xe ra vào

### An ninh và giám sát
- 🔍 **Giám sát an ninh**: Theo dõi phương tiện đáng ngờ
- 📝 **Ghi nhận vi phạm**: Tự động nhận diện biển số vi phạm
- 🚨 **Phản ứng sự cố**: Phát hiện và cảnh báo tai nạn

### Doanh nghiệp
- 🏪 **Quản lý cửa hàng**: Thống kê khách ghé thăm bằng phương tiện
- 📦 **Logistics**: Theo dõi xe tải và phương tiện vận chuyển
- 🏭 **Khu công nghiệp**: Quản lý ra vào phương tiện

## Hướng phát triển tương lai

### Tính năng nâng cao
- 🌐 **Web Interface**: Xây dựng giao diện web để quản lý và theo dõi
- 📱 **Mobile App**: Ứng dụng di động để giám sát từ xa
- ☁️ **Cloud Integration**: Triển khai trên cloud cho khả năng mở rộng
- 🔗 **API Development**: Cung cấp API cho các bên thứ ba

### Cải thiện AI
- 🧠 **Model Enhancement**: Cải thiện độ chính xác phát hiện
- 🚗 **Vehicle Classification**: Phân loại chi tiết hơn (hãng xe, màu sắc)
- 🌍 **Multi-language Support**: Hỗ trợ OCR cho nhiều ngôn ngữ
- 🎯 **Real-time Processing**: Tối ưu cho xử lý thời gian thực

### Tích hợp hệ thống
- 📡 **IoT Integration**: Kết nối với camera và cảm biến IoT
- 🗄️ **Database Integration**: Lưu trữ và quản lý dữ liệu lịch sử
- 📊 **Analytics Dashboard**: Bảng điều khiển phân tích dữ liệu
- 🔔 **Alert System**: Hệ thống cảnh báo thông minh

Dự án được phát triển với mục đích nghiên cứu và học tập.

