
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIGURATION


st.set_page_config(
    page_title="Retail Recommendation Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS


st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .product-title {
        font-size: 22px;
        font-weight: 600;
    }

    .reason-text {
        color: #555555;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #888888;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# LOAD DATA


@st.cache_data
def load_data():
    return pd.read_csv("data/retail_data.csv")



# PREPARE COLLABORATIVE FILTERING MODEL


@st.cache_data
def prepare_model(data):

    user_item_matrix = data.pivot_table(
        index="Customer ID",
        columns="Product Name",
        values="Rating",
        aggfunc="mean"
    ).fillna(0)

    item_user_matrix = user_item_matrix.T

    similarity = cosine_similarity(item_user_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )

    return user_item_matrix, similarity_df



# SAFE DATA LOADING


try:

    data = load_data()

    required_columns = [
        "Customer ID",
        "Product Name",
        "Product Category",
        "Order ID",
        "Quantity",
        "Price",
        "Rating"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    user_item_matrix, item_similarity = prepare_model(data)

except Exception as error:

    st.error(f"Unable to load the dataset: {error}")
    st.stop()



# PRODUCT IMAGE DATABASE


# These are example image URLs.
# Replace them with your own product images if available.

PRODUCT_IMAGES = {

    "Action Figure":
        "https://images.unsplash.com/photo-1594787318286-3d835c1d207f",

    "Lego Set":
        "https://images.unsplash.com/photo-1587654780291-39c9404d746b",

    "Lipstick":
        "https://images.unsplash.com/photo-1586495777744-4413f21062fa",

    "Cookware Set":
        "https://images.unsplash.com/photo-1556911220-e15b29be8c8f",

    "Headphones":
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e",

    "Smartphone":
        "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9",

    "Comics":
        "https://images.unsplash.com/photo-1512820790803-83ca734da794"
}



# CATEGORY IMAGE FALLBACKS


CATEGORY_IMAGES = {

    "Electronics":
        "https://images.unsplash.com/photo-1496181133206-80ce9b88a853",

    "Toys":
        "https://images.unsplash.com/photo-1594787318286-3d835c1d207f",

    "Beauty":
        "https://images.unsplash.com/photo-1596462502278-27bfdc403348",

    "Books":
        "https://images.unsplash.com/photo-1495446815901-a7297e633e8d",

    "Home":
        "https://images.unsplash.com/photo-1556911220-e15b29be8c8f",

    "Clothing":
        "https://images.unsplash.com/photo-1445205170230-053b83016050"
}


def get_product_image(product_name, category):

    if product_name in PRODUCT_IMAGES:
        return PRODUCT_IMAGES[product_name]

    if category in CATEGORY_IMAGES:
        return CATEGORY_IMAGES[category]

    return None



# SIDEBAR


with st.sidebar:

    st.markdown("## 🛒 Retail Engine")

    st.caption(
        "Item-Based Collaborative Filtering"
    )

    st.divider()

    st.subheader("Dashboard Filters")

    categories = sorted(
        data["Product Category"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_category = st.selectbox(
        "Product Category",
        ["All Categories"] + categories
    )

    number_of_recommendations = st.slider(
        "Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    st.divider()

    st.info(
        "The engine recommends products by comparing "
        "similarity between products rated by customers."
    )



# HEADER


st.markdown(
    '<p class="main-title">'
    '🛒 Retail Recommendation Engine'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Personalized product recommendations using '
    'Item-Based Collaborative Filtering'
    '</p>',
    unsafe_allow_html=True
)

st.success(
    f"Dataset loaded successfully! "
    f"{len(data):,} transaction records available."
)



# OVERVIEW METRICS


total_customers = data["Customer ID"].nunique()
total_products = data["Product Name"].nunique()
total_orders = data["Order ID"].nunique()

total_sales = (
    pd.to_numeric(data["Price"], errors="coerce").fillna(0)
    * pd.to_numeric(data["Quantity"], errors="coerce").fillna(0)
).sum()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("👥 Customers", f"{total_customers:,}")

with m2:
    st.metric("🛍️ Products", f"{total_products:,}")

with m3:
    st.metric("📦 Orders", f"{total_orders:,}")

with m4:
    st.metric("💰 Total Sales", f"₹{total_sales:,.2f}")


st.divider()



# CUSTOMER SELECTION


st.header("👤 Customer Recommendation Center")

customer_ids = sorted(
    data["Customer ID"].dropna().unique()
)

default_customer = (
    "CUST1018"
    if "CUST1018" in customer_ids
    else customer_ids[0]
)

customer_id = st.selectbox(
    "Select Customer",
    customer_ids,
    index=customer_ids.index(default_customer)
)

customer_data = data[
    data["Customer ID"] == customer_id
].copy()

customer_purchases = (
    customer_data["Product Name"]
    .dropna()
    .unique()
    .tolist()
)



# CUSTOMER METRICS


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Customer ID", customer_id)

with c2:
    st.metric("Transactions", len(customer_data))

with c3:
    st.metric(
        "Products Purchased",
        len(customer_purchases)
    )

with c4:
    st.metric(
        "Average Rating",
        f"{customer_data['Rating'].mean():.2f}"
    )



# RECOMMENDATION FUNCTION


def recommend_products(
    customer_id,
    n=5,
    category="All Categories"
):

    if customer_id not in user_item_matrix.index:
        return []

    customer_ratings = user_item_matrix.loc[customer_id]

    purchased_items = customer_ratings[
        customer_ratings > 0
    ].index.tolist()

    scores = {}
    reasons = {}

    # --------------------------------------
    # ITEM-BASED COLLABORATIVE FILTERING
    # --------------------------------------

    for purchased_item in purchased_items:

        if purchased_item not in item_similarity.index:
            continue

        similarities = item_similarity.loc[purchased_item]

        for recommended_item, similarity_score in similarities.items():

            if recommended_item in purchased_items:
                continue

            if similarity_score <= 0:
                continue

            if recommended_item not in scores:

                scores[recommended_item] = 0

            scores[recommended_item] += float(
                similarity_score
            )

            # Save the purchased item responsible
            # for the recommendation

            if recommended_item not in reasons:

                reasons[recommended_item] = (
                    purchased_item
                )

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------
    # CATEGORY FILTER
    # --------------------------------------

    if category != "All Categories":

        valid_products = set(
            data[
                data["Product Category"] == category
            ]["Product Name"]
        )

        recommendations = [
            item for item in recommendations
            if item[0] in valid_products
        ]

    # --------------------------------------
    # FALLBACK RECOMMENDATIONS
    # --------------------------------------

    fallback_used = False

    if len(recommendations) < n:

        fallback_used = True

        popular_products = (
            data.groupby("Product Name")
            .agg(
                Average_Rating=("Rating", "mean"),
                Purchase_Count=("Product Name", "size")
            )
            .sort_values(
                ["Average_Rating", "Purchase_Count"],
                ascending=False
            )
        )

        for product, row in popular_products.iterrows():

            if product in purchased_items:
                continue

            if category != "All Categories":

                product_categories = data[
                    data["Product Name"] == product
                ]["Product Category"]

                if category not in product_categories.values:
                    continue

            if product not in dict(recommendations):

                recommendations.append(
                    (
                        product,
                        float(row["Average_Rating"])
                    )
                )

                reasons[product] = (
                    "Popular product with a strong "
                    "average customer rating"
                )

            if len(recommendations) >= n:
                break

    # --------------------------------------
    # CREATE RESULT RECORDS
    # --------------------------------------

    results = []

    for product, score in recommendations[:n]:

        if product in reasons and not fallback_used:

            reason = (
                "Recommended because it is similar "
                f"to your purchased product: "
                f"{reasons[product]}"
            )

        elif product in reasons:

            reason = reasons[product]

        else:

            reason = (
                "Recommended based on product "
                "similarity and customer ratings"
            )

        results.append({
            "Product Name": product,
            "Recommendation Score": round(score, 3),
            "Reason": reason
        })

    return results



# GENERATE RECOMMENDATIONS


if st.button(
    "🎯 Generate Recommendations",
    type="primary",
    width="stretch"
):

    recommendations = recommend_products(
        customer_id=customer_id,
        n=number_of_recommendations,
        category=selected_category
    )

    st.session_state["recommendations"] = recommendations
    st.session_state["recommendation_customer"] = customer_id



# DISPLAY RECOMMENDATIONS


if "recommendations" in st.session_state:

    recommendations = st.session_state["recommendations"]

    recommendation_customer = st.session_state[
        "recommendation_customer"
    ]

    st.header("⭐ Recommended Products")

    st.caption(
        f"Personalized results for "
        f"{recommendation_customer}"
    )

    if recommendations:

        recommendation_data = pd.DataFrame(
            recommendations
        )

        # ----------------------------------
        # ADD PRODUCT INFORMATION
        # ----------------------------------

        product_information = (
            data[
                [
                    "Product Name",
                    "Product Category",
                    "Price"
                ]
            ]
            .drop_duplicates("Product Name")
        )

        recommendation_data = recommendation_data.merge(
            product_information,
            on="Product Name",
            how="left"
        )

        # ----------------------------------
        # DOWNLOAD BUTTON
        # ----------------------------------

        csv_data = recommendation_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv_data,
            file_name=(
                f"recommendations_{recommendation_customer}.csv"
            ),
            mime="text/csv",
            width="stretch"
        )

        st.divider()

        # ----------------------------------
        # RECOMMENDATION CARDS
        # ----------------------------------

        for index, row in recommendation_data.iterrows():

            product_name = row["Product Name"]
            category = row["Product Category"]
            price = row["Price"]
            score = float(row["Recommendation Score"])
            reason = row["Reason"]

            image_url = get_product_image(
                product_name,
                category
            )

            with st.container(border=True):

                col_image, col_info, col_score = st.columns(
                    [1, 4, 2]
                )

                # --------------------------
                # PRODUCT IMAGE
                # --------------------------

                with col_image:

                    if image_url:

                        st.image(
                            image_url,
                            width=120
                        )

                    else:

                        st.markdown(
                            "### 🛍️",
                        )

                # --------------------------
                # PRODUCT DETAILS
                # --------------------------

                with col_info:

                    st.markdown(
                        f"### 🛍️ {product_name}"
                    )

                    st.caption(
                        f"Category: {category}"
                    )

                    st.write(
                        f"💡 **Why recommended:** {reason}"
                    )

                    st.write(
                        f"💰 Price: ₹{price:,.2f}"
                    )

                # --------------------------
                # SCORE DISPLAY
                # --------------------------

                with col_score:

                    st.write(
                        "**Recommendation Score**"
                    )

                    st.metric(
                        "Similarity Score",
                        f"{score:.3f}"
                    )

                    st.caption(
                        "Higher scores indicate stronger "
                        "recommendation relevance."
                    )

        # ----------------------------------
        # SCORE COMPARISON CHART
        # ----------------------------------

        st.subheader(
            "📊 Recommendation Score Comparison"
        )

        chart_data = recommendation_data[
            [
                "Product Name",
                "Recommendation Score"
            ]
        ].set_index("Product Name")

        # Original scores are used here.
        # No artificial 0-to-1 clipping.

        st.bar_chart(
            chart_data,
            horizontal=True
        )

    else:

        st.warning(
            "No recommendations available for "
            "the selected customer and category."
        )



# TABS


tab1, tab2, tab3 = st.tabs(
    [
        "📋 Purchase History",
        "📈 Sales Analytics",
        "🗃️ Dataset Preview"
    ]
)



# PURCHASE HISTORY


with tab1:

    st.subheader("Customer Purchase History")

    if not customer_data.empty:

        history_columns = [
            "Order ID",
            "Product Name",
            "Product Category",
            "Quantity",
            "Price",
            "Rating"
        ]

        st.dataframe(
            customer_data[history_columns],
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "📊 Customer Category Distribution"
        )

        category_counts = (
            customer_data["Product Category"]
            .value_counts()
        )

        if not category_counts.empty:

            st.bar_chart(category_counts)

    else:

        st.info("No purchase history found.")



# SALES ANALYTICS


with tab2:

    st.subheader("📈 Product and Sales Analytics")

    sales_data = data.copy()

    sales_data["Price"] = pd.to_numeric(
        sales_data["Price"],
        errors="coerce"
    ).fillna(0)

    sales_data["Quantity"] = pd.to_numeric(
        sales_data["Quantity"],
        errors="coerce"
    ).fillna(0)

    sales_data["Total Sales"] = (
        sales_data["Price"]
        * sales_data["Quantity"]
    )

    st.subheader("Top 10 Products by Sales")

    top_products = (
        sales_data.groupby("Product Name")["Total Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_products)

    st.subheader("Average Rating by Category")

    category_ratings = (
        data.groupby("Product Category")["Rating"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_ratings)

    st.subheader("Sales by Product Category")

    category_sales = (
        sales_data.groupby("Product Category")["Total Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_sales)



# DATASET PREVIEW


with tab3:

    st.subheader("🗃️ Complete Dataset")

    st.write(
        f"Showing {len(data):,} records and "
        f"{len(data.columns)} columns."
    )

    st.dataframe(
        data,
        width="stretch",
        hide_index=True
    )



# FOOTER


st.divider()

st.markdown(
    '<div class="footer">'
    'Retail Recommendation Engine | '
    'TY B.Sc. Data Science Project | '
    'Item-Based Collaborative Filtering'
    '</div>',
    unsafe_allow_html=True
)