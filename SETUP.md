# Setup dự án trên Google Colab

## Notebooks và Google Drive

Phần này hướng dẫn chuẩn bị thư mục, đặt dữ liệu EMOTIC trên Google Drive và chạy mã trong Google Colab. Giữ dữ liệu gốc trong Drive; không đưa dữ liệu EMOTIC hoặc checkpoint lên GitHub.

### 1. Cấu trúc thư mục

Trong repo, dùng cấu trúc sau:

```text
project-root/
├── configs/
├── data/
│   ├── raw/
│   │   └── emotic/       # Dữ liệu gốc, không sửa
│   ├── processed/        # Dữ liệu đã xử lý hoặc metadata tạo ra
│   └── samples/          # Một tập nhỏ để kiểm tra nhanh
├── notebooks/
│   └── 01_inspect_emotic.ipynb
├── src/
│   └── datasets/
│       └── emotic_dataset.py
├── outputs/              # Kết quả sinh ra khi chạy
└── requirements.txt
```

Trên Google Drive, tạo thư mục riêng cho dữ liệu và kết quả. Ví dụ:

```text
MyDrive/
└── contextual-emotion-reasoning/
    ├── data/
    │   ├── raw/
    │   │   └── emotic/
    │   ├── processed/
    │   └── samples/
    └── outputs/
```

Giải nén bộ EMOTIC vào `data/raw/emotic/`, giữ nguyên cấu trúc và tên file của bộ dữ liệu. Các thư mục annotation và ảnh có thể khác nhau tùy phiên bản EMOTIC; ghi lại vị trí thực tế để truyền vào cấu hình hoặc notebook. Không chỉnh sửa bản gốc. Nếu cần tạo ảnh crop hay metadata đã chuẩn hóa, lưu chúng trong `data/processed/`.

### 2. Mở notebook trong Colab và gắn Google Drive

Có thể mở notebook từ GitHub bằng **Google Colab → File → Open notebook → GitHub**, rồi nhập URL repo. Hoặc mở notebook đã tải lên Drive. Trong notebook, chạy ô sau để gắn Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Colab sẽ yêu cầu đăng nhập và cấp quyền truy cập Drive. Đặt đường dẫn dữ liệu và thư mục kết quả:

```python
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/contextual-emotion-reasoning')
DATA_ROOT = DRIVE_ROOT / 'data'
EMOTIC_ROOT = DATA_ROOT / 'raw' / 'emotic'
OUTPUT_ROOT = DRIVE_ROOT / 'outputs'

for path in (EMOTIC_ROOT, DATA_ROOT / 'processed', DATA_ROOT / 'samples', OUTPUT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

print('EMOTIC:', EMOTIC_ROOT)
print('Tồn tại:', EMOTIC_ROOT.exists())
```

Nếu bạn lưu dữ liệu ở vị trí khác, hãy sửa `DRIVE_ROOT` cho đúng. Kiểm tra một số file bên trong để chắc rằng Drive đã gắn đúng và dữ liệu đã được giải nén:

```python
files = [p for p in EMOTIC_ROOT.rglob('*') if p.is_file()]
print(f'Số file: {len(files)}')
for path in files[:20]:
    print(path.relative_to(EMOTIC_ROOT))
```

Nếu số file là `0`, kiểm tra lại đường dẫn, quyền truy cập Drive và việc giải nén dữ liệu.

### 3. Lấy mã nguồn vào Colab

Nên chạy mã nguồn trong `/content` của Colab và dùng Drive để lưu dữ liệu cùng kết quả. Cách này tránh việc đọc và ghi mã nguồn trực tiếp trên Drive bị chậm. Thay URL dưới đây bằng URL repo của bạn:

```bash
!git clone https://github.com/<username>/<repository>.git /content/contextual-emotion-reasoning
%cd /content/contextual-emotion-reasoning
```

Nếu repo đã được clone trong phiên Colab hiện tại, chỉ cần chạy `%cd` tới thư mục đó. Mỗi khi runtime Colab bị khởi động lại, `/content` có thể bị xóa; clone lại repo, sau đó mount Drive và đặt lại các biến đường dẫn.

### 4. Cài thư viện

Trong Colab, chọn **Runtime → Change runtime type → T4 GPU** nếu notebook cần GPU. Sau đó cài các thư viện trong repo:

```bash
%cd /content/contextual-emotion-reasoning
!pip install -r requirements.txt
```

Nếu dự án chưa có `requirements.txt`, hãy tạo file này và liệt kê các thư viện thực sự được mã nguồn sử dụng trước khi chạy ô cài đặt.

### 5. Chạy notebook kiểm tra dữ liệu

Mở `notebooks/01_inspect_emotic.ipynb` trong Colab. Cập nhật notebook để dùng `EMOTIC_ROOT` ở trên, rồi kiểm tra lần lượt:

1. Đọc được annotation và tìm thấy đường dẫn ảnh tương ứng.
2. Đọc và hiển thị được ảnh toàn cảnh.
3. Bounding box nằm đúng vị trí và crop được người mục tiêu.
4. Đọc đúng các nhãn cảm xúc và giá trị VAD.
5. Kiểm tra mẫu có nhiều người để xác nhận mỗi annotation gắn đúng bounding box.

Chỉ chuyển sang huấn luyện baseline sau khi các kiểm tra này đúng trên nhiều mẫu. Notebook dùng để khám phá và kiểm tra nhanh; phần huấn luyện chính nên chạy bằng script Python để dễ lặp lại.

### 6. Lưu kết quả về Drive

Khi tạo file kết quả trong notebook, ghi chúng vào Drive để không mất sau khi runtime kết thúc:

```python
from datetime import datetime

run_dir = OUTPUT_ROOT / 'notebooks' / datetime.now().strftime('%Y%m%d-%H%M%S')
run_dir.mkdir(parents=True, exist_ok=True)
print('Kết quả phiên này sẽ lưu tại:', run_dir)
```

Lưu log, biểu đồ, dự đoán và checkpoint vào các thư mục con phù hợp dưới `OUTPUT_ROOT`. Không lưu các file lớn hoặc dữ liệu riêng tư vào repo nếu chúng không cần thiết để chạy mã.
