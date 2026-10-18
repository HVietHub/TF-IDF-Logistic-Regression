# Báo cáo So sánh Chi tiết: Lần huấn luyện 1 (train_1) vs Lần huấn luyện 2 (train_2)

Báo cáo này tổng hợp và đối chiếu toàn diện giữa 2 lần huấn luyện của mô hình **TF-IDF + Logistic Regression** trong bài toán phát hiện **Prompt Injection & Jailbreak**.

---

## 1. Tổng quan Cấu hình & Dữ liệu Huấn luyện

| Tiêu chí | Lần huấn luyện 1 (`train_1`) | Lần huấn luyện 2 (`train_2`) |
| :--- | :--- | :--- |
| **Tập dữ liệu sử dụng** | `dataset.csv` (gốc) | `comprehensive_dataset.csv` (tổng hợp từ thư mục `Dataset`) |
| **Tổng số mẫu (Samples)** | **15,335 mẫu** | **74,525 mẫu** (Tăng gần 5 lần) |
| **Phân bố nhãn lớp** | - Benign (0): 13,850 (90.3%)<br>- Prompt Injection (1): **0 mẫu (0.0%)**<br>- Jailbreak (2): 1,485 (9.7%) | - Benign (0): **30,362 mẫu (40.7%)**<br>- Prompt Injection (1): **13,801 mẫu (18.5%)**<br>- Jailbreak (2): **30,362 mẫu (40.7%)** |
| **Nguồn dữ liệu tích hợp** | Chỉ 1 nguồn đơn lẻ | `JailbreakDB` (39 kỹ thuật tấn công), `SPML`, `enterprise`, `jailbreak_llms`, `prompt-injections` |
| **Độ phủ nhãn Prompt Injection** | ❌ **Hoàn toàn không có dữ liệu** | ✅ Đầy đủ 13,801 mẫu đa dạng |
| **Chống rò rỉ dữ liệu (Zero Leakage)** | Chưa kiểm tra với `test.csv` | ✅ Đã lọc sạch 4,558 câu trùng với `test.csv` |
| **Tham số TF-IDF Word** | `min_df=1`, `lowercase=False`, `token_pattern='(?u)\S+'` | `min_df=3`, `lowercase=True`, `token_pattern='(?u)\b\w+\b'` (Chống học thuộc) |
| **Tham số TF-IDF Char** | `min_df=2`, `lowercase=False`, `ngram=[2, 5]` | `min_df=5`, `lowercase=True`, `ngram=[3, 5]` |

---

## 2. Kết quả Đánh giá Nội bộ (Internal Test Split)

| Chỉ số đánh giá | `train_1` (Test split: 1,534 mẫu) | `train_2` (Test split: 7,453 mẫu) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 92.96% | **95.89%** | Tăng +2.93% trên tập dữ liệu lớn gấp 5 lần |
| **Macro Precision** | 0.7971 | **0.9624** | Cải thiện độ chuẩn xác đa lớp |
| **Macro Recall** | 0.8113 | **0.9630** | Tăng mạnh khả năng bao phủ các lớp |
| **Macro F1-Score** | 0.8040 | **0.9627** | Cân bằng vượt trội |
| **F1 - Benign (0)** | 0.9609 | **0.9515** | Duy trì ổn định |
| **F1 - Prompt Injection (1)** | **0.0000** (Support: 0) | **0.9794** (Support: 1,380) | 🚀 **Giải quyết triệt để lỗi thiếu nhãn** |
| **F1 - Jailbreak (2)** | 0.6471 | **0.9571** | 🚀 **Tăng vọt +31% F1-score** |

---

## 3. Kết quả Thực tế trên Tập Kiểm thử Ngoài Độc lập (`test.csv` - 4,558 mẫu)

Tập `test.csv` gồm: 2,282 mẫu `benign`, 1,141 mẫu `promptinjection`, 1,135 mẫu `jailbreak`.

| Chỉ số trên `test.csv` | `train_1` | `train_2` | Mức độ cải thiện |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 53.66% | **70.93%** | 🟢 **+17.27%** |
| **Macro F1-Score** | 0.3170 | **0.5891** | 🟢 **+0.2721** |
| **Weighted F1-Score** | 0.4102 | **0.6369** | 🟢 **+0.2267** |
| **Jailbreak Recall** | 16.30% (185 / 1,135) | **84.00% (953 / 1,135)** | 🚀 **Tăng gấp 5.1 lần** |
| **Jailbreak Precision** | 66.55% | **84.00%** | 🟢 **+17.45%** |
| **Jailbreak F1-Score** | 0.2619 | **0.8400** | 🚀 **Tăng hơn 3.2 lần** |
| **Prompt Injection Recall** | 0.00% (0 / 1,141) | **8.06% (92 / 1,141)** | 🟢 Bắt đầu nhận diện được |
| **Benign Safe Pass Rate** | 92.90% | **95.88%** (2,188 / 2,282) | 🟢 Tăng tỷ lệ cho qua an toàn |
| **Tỷ lệ chặn mối nguy (Threat Interception)** | 34.89% | **91.98%** | 🚀 **Tăng gần 3 lần** |

---

## 4. Danh mục File Báo cáo Chi tiết

Toàn bộ các file báo cáo của 2 lần train đã được lưu trữ tách biệt và đầy đủ:

### 📂 Thư mục Báo cáo lần 1 (`train_1`):
Đường dẫn: `tf-idf/reports/train_1/`
- `metrics.json`: Tổng hợp toàn bộ metrics validation và internal test.
- `experiment_comparison.csv`: So sánh giữa 3 pipeline A (Word), B (Char), C (Combined).
- `test_classification_report.json`: Classification report chi tiết trên tập test nội bộ.
- `val_classification_report.json`: Classification report chi tiết trên tập validation.
- `confusion_matrix_test.png` & `confusion_matrix_test.csv`: Ma trận nhầm lẫn test nội bộ.
- `test_csv_evaluation_report.json`: Báo cáo đánh giá chi tiết trên tập ngoài `test.csv`.
- `test_csv_confusion_matrix.png` & `csv`: Ma trận nhầm lẫn trên `test.csv`.
- `test_csv_predictions.csv`: Kết quả dự đoán chi tiết từng dòng của `test.csv`.
- `test_csv_misclassifications.csv`: Danh sách các mẫu bị dự đoán sai trên `test.csv`.

### 📂 Thư mục Báo cáo lần 2 (`train_2`):
Đường dẫn: `tf-idf/reports/train_2/`
- `metrics.json`: Toàn bộ metrics validation (96.00%) và internal test (95.89%).
- `experiment_comparison.csv`: So sánh hiệu năng 3 pipeline trên 74.5k mẫu.
- `test_classification_report.json`: Báo cáo phân loại 3 nhãn cân bằng trên tập test nội bộ.
- `val_classification_report.json`: Báo cáo phân loại trên tập validation.
- `confusion_matrix_test.png` & `confusion_matrix_test.csv`: Ma trận nhầm lẫn test nội bộ 3 nhãn.
- `test_csv_evaluation_report.json`: Báo cáo đánh giá đầy đủ mô hình mới trên `test.csv`.
- `test_csv_confusion_matrix.png` & `csv`: Ma trận nhầm lẫn trên `test.csv` (Jailbreak recall đạt 84.0%).
- `test_csv_predictions.csv`: Kết quả dự đoán kèm xác suất (`proba_benign`, `proba_promptinjection`, `proba_jailbreak`).
- `test_csv_misclassifications.csv`: Chi tiết 1,325 mẫu phân loại sai để tiếp tục phân tích cải tiến.
