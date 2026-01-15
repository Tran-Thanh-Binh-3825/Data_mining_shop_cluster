import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==================================================
# CẤU HÌNH TRANG
# ==================================================
st.set_page_config(
    page_title="Dashboard Phân Khúc Khách Hàng",
    layout="wide"
)

# ==================================================
# ĐƯỜNG DẪN DỮ LIỆU
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTER_PATH = os.path.join(
    BASE_DIR, "data", "processed", "customer_clusters_from_rules.csv"
)
RULES_PATH = os.path.join(
    BASE_DIR, "data", "processed", "rules_apriori_filtered.csv"
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    clusters = pd.read_csv(CLUSTER_PATH)
    rules = pd.read_csv(RULES_PATH)
    return clusters, rules

clusters_df, rules_df = load_data()

# ==================================================
# SIDEBAR – BỘ LỌC
# ==================================================
st.sidebar.title("🔎 Bộ lọc")

cluster_list = sorted(clusters_df["cluster"].unique())
selected_cluster = st.sidebar.selectbox(
    "Chọn cụm khách hàng",
    cluster_list
)

# ==================================================
# TIÊU ĐỀ
# ==================================================
st.title("🛒 DASHBOARD PHÂN KHÚC KHÁCH HÀNG")
st.markdown(
    """
    Dashboard hỗ trợ **phân tích cụm khách hàng** dựa trên  
    **Luật mua kèm (Association Rules) kết hợp với RFM**.
    """
)

# ==================================================
# TỔNG QUAN CỤM
# ==================================================
st.header(f"📊 Tổng quan Cụm {selected_cluster}")

cluster_data = clusters_df[
    clusters_df["cluster"] == selected_cluster
]

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Số khách hàng", len(cluster_data))

if "Recency" in cluster_data.columns:
    col2.metric("⏱️ Recency TB", round(cluster_data["Recency"].mean(), 1))

if "Frequency" in cluster_data.columns:
    col3.metric("🔁 Frequency TB", round(cluster_data["Frequency"].mean(), 1))

if "Monetary" in cluster_data.columns:
    col4.metric("💰 Monetary TB", round(cluster_data["Monetary"].mean(), 0))

# ==================================================
# DANH SÁCH KHÁCH HÀNG
# ==================================================
st.subheader("👥 Danh sách khách hàng trong cụm")

st.dataframe(
    cluster_data.head(100),
    use_container_width=True
)

# ==================================================
# LUẬT MUA KÈM NỔI BẬT
# ==================================================
st.header("🔗 Các luật mua kèm tiêu biểu")

top_rules = (
    rules_df
    .sort_values(["lift", "confidence"], ascending=False)
    .head(10)
    [["antecedents", "consequents", "support", "confidence", "lift"]]
)

st.dataframe(
    top_rules.rename(columns={
        "antecedents": "Sản phẩm mua trước",
        "consequents": "Sản phẩm mua kèm",
        "support": "Support",
        "confidence": "Confidence",
        "lift": "Lift"
    }),
    use_container_width=True
)

# ==================================================
# GỢI Ý BUNDLE / CROSS-SELL
# ==================================================
st.header("🎯 Gợi ý Bundle / Cross-sell")

bundle_df = top_rules.copy()
bundle_df["Gợi ý"] = (
    "Gợi ý bán kèm: "
    + bundle_df["antecedents"].astype(str)
    + " → "
    + bundle_df["consequents"].astype(str)
)

st.table(
    bundle_df[["Gợi ý", "confidence", "lift"]]
    .rename(columns={
        "confidence": "Độ tin cậy",
        "lift": "Độ mạnh luật"
    })
)

# ==================================================
# CHIẾN LƯỢC MARKETING GỢI Ý
# ==================================================
st.header("📣 Chiến lược Marketing đề xuất")

if "Recency" in cluster_data.columns and cluster_data["Recency"].mean() > 100:
    st.success(
        "🔔 **Chiến dịch kích hoạt lại khách hàng**  \n"
        "Gửi email nhắc mua, voucher có thời hạn để kéo khách quay lại."
    )
elif (
    "Monetary" in cluster_data.columns
    and cluster_data["Monetary"].mean() > clusters_df["Monetary"].mean()
):
    st.success(
        "💎 **Upsell & Bundle cao cấp**  \n"
        "Thiết kế combo giá trị cao, ưu đãi VIP cho nhóm khách hàng chi tiêu lớn."
    )
else:
    st.success(
        "🛍️ **Chiến lược Cross-sell**  \n"
        "Gợi ý sản phẩm bổ trợ dựa trên các luật mua kèm phổ biến."
    )

# ==================================================
# TRỰC QUAN PHÂN CỤM (PCA)
# ==================================================
st.header("🧭 Trực quan phân cụm (PCA 2D)")

rfm_cols = [c for c in ['Recency', 'Frequency', 'Monetary'] if c in clusters_df.columns]
feature_cols = rfm_cols

X = clusters_df[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(
    Z[:, 0],
    Z[:, 1],
    c=clusters_df['cluster'],
    cmap='viridis',
    s=15
)

ax.set_xlabel("Thành phần chính 1")
ax.set_ylabel("Thành phần chính 2")
ax.set_title("Biểu đồ phân cụm khách hàng (PCA)")

legend = ax.legend(
    *scatter.legend_elements(),
    title="Cụm"
)
ax.add_artist(legend)

st.pyplot(fig)

# ==================================================
# DỮ LIỆU GỐC
# ==================================================
with st.expander("📄 Xem dữ liệu chi tiết"):
    st.dataframe(clusters_df, use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("Mini Project Data Mining – Phân khúc khách hàng dựa trên luật mua kèm")
