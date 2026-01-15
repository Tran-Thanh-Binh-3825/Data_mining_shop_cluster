import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Customer Clustering Dashboard",
    layout="wide"
)
import streamlit as st
import pandas as pd

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

CLUSTER_PATH = "..data/processed/customer_clusters_from_rules.csv"
RULES_PATH = "..data/processed/rules_apriori_filtered.csv"

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    clusters = pd.read_csv(CLUSTER_PATH)
    rules = pd.read_csv(RULES_PATH)
    return clusters, rules

clusters_df, rules_df = load_data()

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("🔎 Filters")

cluster_list = sorted(clusters_df["cluster"].unique())
selected_cluster = st.sidebar.selectbox(
    "Select cluster",
    cluster_list
)

# ==============================
# HEADER
# ==============================
st.title("🛒 Customer Segmentation & Rule-based Insights")
st.markdown(
    "Dashboard for exploring customer clusters, "
    "association rules, and marketing strategies."
)

# ==============================
# CLUSTER OVERVIEW
# ==============================
st.header(f"📊 Cluster {selected_cluster} Overview")

cluster_data = clusters_df[
    clusters_df["cluster"] == selected_cluster
]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", len(cluster_data))

if "Recency" in cluster_data.columns:
    col2.metric("Avg Recency", round(cluster_data["Recency"].mean(), 1))

if "Frequency" in cluster_data.columns:
    col3.metric("Avg Frequency", round(cluster_data["Frequency"].mean(), 1))

if "Monetary" in cluster_data.columns:
    col4.metric("Avg Monetary", round(cluster_data["Monetary"].mean(), 1))

# ==============================
# CUSTOMER TABLE
# ==============================
st.subheader("👥 Customers in this cluster")

st.dataframe(
    cluster_data.head(100),
    use_container_width=True
)

# ==============================
# TOP RULES FOR CLUSTER
# ==============================
st.header("🔗 Top Association Rules")

# Heuristic: rules with higher lift & confidence
top_rules = (
    rules_df
    .sort_values(["lift", "confidence"], ascending=False)
    .head(10)
    [["antecedents", "consequents", "support", "confidence", "lift"]]
)

st.dataframe(top_rules, use_container_width=True)

# ==============================
# BUNDLE / CROSS-SELL SUGGESTION
# ==============================
st.header("🎯 Bundle & Cross-sell Suggestions")

bundle_df = top_rules.copy()
bundle_df["Suggestion"] = (
    "Bundle " + bundle_df["antecedents"].astype(str)
    + " → " + bundle_df["consequents"].astype(str)
)

st.table(bundle_df[["Suggestion", "confidence", "lift"]])

# ==============================
# MARKETING STRATEGY (TEXT)
# ==============================
st.header("📣 Suggested Marketing Strategy")

if "Recency" in cluster_data.columns and cluster_data["Recency"].mean() > 100:
    st.success(
        "🔔 **Win-back campaign**: "
        "Offer time-limited vouchers and reminder emails."
    )
elif "Monetary" in cluster_data.columns and cluster_data["Monetary"].mean() > clusters_df["Monetary"].mean():
    st.success(
        "💎 **Premium bundle & upsell**: "
        "Create high-value product bundles and loyalty rewards."
    )
else:
    st.success(
        "🛍️ **Cross-sell strategy**: "
        "Recommend complementary products based on association rules."
    )

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Data Mining Mini Project – Rule-based Customer Segmentation")

# ======================
# PATH FIX (QUAN TRỌNG)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_cluster_result.csv"
)

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

st.write("📂 Data path:", DATA_PATH)
st.write("✅ File tồn tại:", os.path.exists(DATA_PATH))

df = load_data(DATA_PATH)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("🔍 Bộ lọc")

clusters = sorted(df['cluster'].unique())
selected_clusters = st.sidebar.multiselect(
    "Chọn cụm",
    clusters,
    default=clusters
)

df = df[df['cluster'].isin(selected_clusters)]

# ======================
# TITLE
# ======================
st.title("📊 CUSTOMER SEGMENTATION DASHBOARD")
st.markdown(
    """
    Phân cụm khách hàng dựa trên **Luật mua sắm (Association Rules) + RFM**
    """
)

# ======================
# OVERVIEW
# ======================
st.subheader("🔢 Tổng quan")

col1, col2, col3 = st.columns(3)
col1.metric("Số khách hàng", df['CustomerID'].nunique())
col2.metric("Số cụm", df['cluster'].nunique())

if 'Monetary' in df.columns:
    col3.metric("Chi tiêu TB", f"{df['Monetary'].mean():,.0f}")

# ======================
# CLUSTER DISTRIBUTION
# ======================
st.subheader("📌 Phân bố khách hàng theo cụm")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='cluster', palette='viridis', ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Số khách")
st.pyplot(fig)

# ======================
# RFM PROFILING
# ======================
st.subheader("🧠 Hồ sơ RFM theo cụm")

rfm_cols = [c for c in ['Recency', 'Frequency', 'Monetary'] if c in df.columns]

rfm_summary = (
    df.groupby('cluster')[rfm_cols]
    .mean()
    .round(2)
)

st.dataframe(rfm_summary, use_container_width=True)

# ======================
# RULE ANALYSIS
# ======================
st.subheader("🛒 Luật mua sắm nổi bật")

rule_cols = [c for c in df.columns if c.startswith("Rule_")]

for cluster_id in sorted(df['cluster'].unique()):
    st.markdown(f"### 🔹 Cụm {cluster_id}")

    cluster_df = df[df['cluster'] == cluster_id]
    rule_means = cluster_df[rule_cols].mean().sort_values(ascending=False)

    top_rules = rule_means[rule_means > 0.1].head(5)

    if len(top_rules) == 0:
        st.write("Không có luật nổi bật.")
    else:
        for rule, val in top_rules.items():
            st.write(f"- **{rule}**: {val:.3f}")

# ======================
# PCA VISUALIZATION
# ======================
st.subheader("🧭 Trực quan 2D (PCA)")

feature_cols = rfm_cols + rule_cols
X = df[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(
    Z[:, 0],
    Z[:, 1],
    c=df['cluster'],
    cmap='viridis',
    s=15
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Customer Clusters (PCA)")

legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)

st.pyplot(fig)

# ======================
# RAW DATA
# ======================
with st.expander("📄 Xem dữ liệu chi tiết"):
    st.dataframe(df, use_container_width=True)
