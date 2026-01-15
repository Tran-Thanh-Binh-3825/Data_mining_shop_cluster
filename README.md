# 📊 Mini Project: Rule-based Customer Segmentation

## 1. Giới thiệu
Trong mini project này, nhóm xây dựng một pipeline phân khúc khách hàng dựa trên hướng tiếp cận:

**Luật kết hợp → Đặc trưng hành vi mua kèm → Phân cụm → Diễn giải → Đề xuất chiến lược marketing**

Mục tiêu là khai thác các **association rules** (Apriori / FP-Growth) để biểu diễn hành vi mua sắm của khách hàng, kết hợp với **RFM (Recency – Frequency – Monetary)** nhằm tạo ra các cụm khách hàng có ý nghĩa và có thể ứng dụng trong marketing.

---

## 2. Dữ liệu
- Dữ liệu giao dịch bán lẻ đã được làm sạch:  
  `data/processed/cleaned_uk_data.csv`
- Mỗi bản ghi gồm các trường chính:
  - `CustomerID`
  - `InvoiceDate`
  - `StockCode`
  - `Quantity`
  - `UnitPrice`
- Dữ liệu được xử lý để tạo:
  - Ma trận **Customer × Item**
  - Bộ **luật kết hợp (association rules)**

---

## 3. Khai phá luật kết hợp (Association Rules)

### 3.1 Thuật toán
- Sử dụng **Apriori** (hoặc FP-Growth)
- Dạng luật:

### 3.2 Chiến lược chọn luật
- Chỉ sử dụng **Top-K luật** (ví dụ: Top 200)
- Luật được sắp xếp theo:
- `lift` (ưu tiên phản ánh độ mạnh của mối quan hệ)
- Không sử dụng toàn bộ luật để tránh:
- Nhiễu
- Không gian đặc trưng quá lớn

### 3.3 Ví dụ các luật tiêu biểu

| Antecedents | Consequents | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| {A} | {B} | 0.021 | 0.42 | 2.1 |
| {C} | {D} | 0.018 | 0.38 | 1.9 |
| {E, F} | {G} | 0.012 | 0.55 | 2.5 |

(Các luật trên được trích từ tập luật đầu vào dùng cho phân cụm)

---

## 4. Feature Engineering cho phân cụm

Nhóm xây dựng **nhiều biến thể đặc trưng** để so sánh.

### 4.1 Biến thể 1 – Baseline (Rule Binary)
- Mỗi luật là một đặc trưng nhị phân
- Giá trị = 1 nếu khách hàng thỏa antecedents của luật

### 4.2 Biến thể 2 – Rule + Weighting / RFM
- Gán trọng số cho rule-feature (ví dụ: `lift`, `lift × confidence`)
- Kết hợp thêm **RFM**:
- Recency
- Frequency
- Monetary
- Có tùy chọn scale RFM và rule-feature

### 4.3 Mục tiêu so sánh
- Rule-only vs Rule + RFM
- Binary vs Weighted rules
- Top-K nhỏ vs Top-K lớn

---

## 5. Phân cụm khách hàng

### 5.1 Thuật toán
- Sử dụng **K-Means**

### 5.2 Chọn số cụm K
- Khảo sát K trong khoảng **2 → 10**
- Sử dụng **Silhouette Score**
- Chọn K tối ưu dựa trên:
- Điểm silhouette
- Khả năng diễn giải và ứng dụng marketing

### 5.3 Kết quả
- Mỗi khách hàng được gán nhãn `cluster`
- Kết quả được lưu tại:

---

## 6. Đánh giá và trực quan hóa

- Giảm chiều về 2D bằng **PCA**
- Vẽ scatter plot, tô màu theo cluster
- Nhận xét mức độ:
- Tách cụm
- Chồng lấn giữa các cụm

Biểu đồ giúp đánh giá trực quan chất lượng phân cụm.

---

## 7. Profiling & Diễn giải cụm

### 7.1 Thống kê theo cụm
- Số lượng khách hàng
- Trung bình Recency – Frequency – Monetary
- Các luật mua kèm được kích hoạt nhiều nhất trong cụm

### 7.2 Đặt tên & Persona (ví dụ)

| Cluster | Tên (EN) | Tên (VI) |
|-------|----------|----------|
| 0 | High-Value Bundlers | Khách mua combo giá trị cao |
| 1 | Dormant Customers | Khách hàng ngủ đông |
| 2 | Category-focused Buyers | Khách trung thành theo danh mục |

### 7.3 Chiến lược marketing
- **High-Value Bundlers**: bundle sản phẩm, upsell
- **Dormant Customers**: win-back campaign, voucher quay lại
- **Category-focused Buyers**: cross-sell trong cùng danh mục

Các chiến lược được đề xuất **trực tiếp dựa trên đặc trưng RFM và các luật mua kèm** của từng cụm.

---

## 8. Dashboard Streamlit

Nhóm xây dựng **dashboard bằng Streamlit** để:
- Lọc và xem từng cụm khách hàng
- Xem hồ sơ RFM
- Xem các luật mua kèm tiêu biểu
- Gợi ý bundle / cross-sell và chiến lược marketing

### Chạy dashboard
```bash
streamlit run app.py
