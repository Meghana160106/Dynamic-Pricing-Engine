import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Dynamic Pricing Engine")

st.markdown("""
### Machine Learning Based Dynamic Pricing and Demand Optimization

This system analyzes e-commerce data, demand, inventory,
competitor pricing and discounts to recommend an optimized
selling price.
""")

def generate_demo_data(n=1500):

    np.random.seed(42)

    df = pd.DataFrame({
        "Product_ID": range(1, n + 1),
        "Base_Price": np.random.uniform(100, 3000, n),
        "Competitor_Price": np.random.uniform(100, 3200, n),
        "Inventory": np.random.randint(10, 500, n),
        "Sales": np.random.randint(10, 1000, n),
        "Discount": np.random.uniform(0, 40, n),
        "Rating": np.random.uniform(2.5, 5, n)
    })

    return df

def find_column(df, names):

    column_map = {}

    for column in df.columns:

        key = str(column).lower()
        key = key.replace(" ", "_")
        key = key.replace("-", "_")

        column_map[key] = column

    for name in names:

        key = name.lower()
        key = key.replace(" ", "_")
        key = key.replace("-", "_")

        if key in column_map:
            return column_map[key]

    for column in df.columns:

        column_name = str(column).lower()

        for name in names:

            if name.lower() in column_name:

                return column

    return None

st.sidebar.header("📂 Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or ZIP",
    type=["csv", "zip"]
)

data = None

if uploaded_file is None:

    st.info(
        "No dataset uploaded. Demo e-commerce data is being used."
    )

    data = generate_demo_data()

else:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            data = pd.read_csv(uploaded_file)

        elif uploaded_file.name.lower().endswith(".zip"):

            zip_bytes = uploaded_file.read()

            with zipfile.ZipFile(
                io.BytesIO(zip_bytes)
            ) as zip_file:

                csv_files = [
                    name
                    for name in zip_file.namelist()
                    if name.lower().endswith(".csv")
                    and not name.startswith("__MACOSX")
                ]

                if len(csv_files) == 0:

                    st.error(
                        "No CSV file was found inside the ZIP."
                    )

                    st.stop()

                selected_csv = st.sidebar.selectbox(
                    "Select CSV from ZIP",
                    csv_files
                )

                with zip_file.open(
                    selected_csv
                ) as file:

                    data = pd.read_csv(file)

    except Exception as error:

        st.error(
            f"Unable to read dataset: {error}"
        )

        st.stop()

data = data.copy()

data.columns = [
    str(column).strip()
    for column in data.columns
]

data = data.dropna(
    axis=0,
    how="all"
)

data = data.dropna(
    axis=1,
    how="all"
)

data = data.reset_index(
    drop=True
)

price_col = find_column(
    data,
    [
        "price",
        "base_price",
        "selling_price",
        "unit_price",
        "mrp",
        "sale_price"
    ]
)

competitor_col = find_column(
    data,
    [
        "competitor_price",
        "competitor",
        "market_price"
    ]
)

inventory_col = find_column(
    data,
    [
        "inventory",
        "stock",
        "stock_quantity",
        "quantity_in_stock"
    ]
)

sales_col = find_column(
    data,
    [
        "sales",
        "units_sold",
        "quantity_sold",
        "demand",
        "orders",
        "units"
    ]
)

discount_col = find_column(
    data,
    [
        "discount",
        "discount_percent",
        "discount_percentage"
    ]
)

detected_columns = [
    price_col,
    competitor_col,
    inventory_col,
    sales_col,
    discount_col
]

for column in detected_columns:

    if column is not None:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

if price_col is None:

    numeric_columns = data.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) > 0:

        price_col = numeric_columns[0]

    else:

        data["Base_Price"] = 1000.0

        price_col = "Base_Price"

if sales_col is None:

    np.random.seed(42)

    data["Sales"] = np.random.randint(
        20,
        500,
        len(data)
    )

    sales_col = "Sales"

if inventory_col is None:

    np.random.seed(10)

    data["Inventory"] = np.random.randint(
        20,
        500,
        len(data)
    )

    inventory_col = "Inventory"

if competitor_col is None:

    price_values = pd.to_numeric(
        data[price_col],
        errors="coerce"
    )

    median_price = price_values.median()

    if pd.isna(median_price):

        median_price = 1000

    price_values = price_values.fillna(
        median_price
    )

    np.random.seed(20)

    data["Competitor_Price"] = (
        price_values *
        np.random.uniform(
            0.90,
            1.10,
            len(data)
        )
    )

    competitor_col = "Competitor_Price"

if discount_col is None:

    data["Discount"] = 10.0

    discount_col = "Discount"

important_columns = [
    price_col,
    competitor_col,
    inventory_col,
    sales_col,
    discount_col
]

for column in important_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

for column in important_columns:

    median_value = data[column].median()

    if pd.isna(median_value):

        median_value = 0

    data[column] = data[column].fillna(
        median_value
    )

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.fillna(0)

data = data.reset_index(
    drop=True
)

data[price_col] = data[price_col].clip(
    lower=1
)

data[competitor_col] = data[
    competitor_col
].clip(
    lower=1
)

data[inventory_col] = data[
    inventory_col
].clip(
    lower=1
)

data[sales_col] = data[
    sales_col
].clip(
    lower=0
)

data[discount_col] = data[
    discount_col
].clip(
    lower=0,
    upper=100
)

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Records",
    f"{len(data):,}"
)

col2.metric(
    "Average Price",
    f"₹{data[price_col].mean():,.2f}"
)

col3.metric(
    "Average Demand",
    f"{data[sales_col].mean():,.0f}"
)

col4.metric(
    "Average Inventory",
    f"{data[inventory_col].mean():,.0f}"
)

with st.expander("🔍 View Dataset"):

    st.dataframe(
        data.head(50),
        use_container_width=True,
        hide_index=True
    )

features = [
    price_col,
    competitor_col,
    inventory_col,
    discount_col
]

X = data[features].copy()

y = data[sales_col].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.fillna(
    X.median()
)

y = pd.to_numeric(
    y,
    errors="coerce"
)

y = y.fillna(
    y.median()
)

model = None
mae = 0
r2 = 0

if len(data) >= 20:

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

    except Exception:

        model = None

st.subheader("🤖 Demand Prediction Model")

m1, m2 = st.columns(2)

m1.metric(
    "Mean Absolute Error",
    f"{mae:.2f}"
)

m2.metric(
    "R² Score",
    f"{r2:.3f}"
)

st.sidebar.divider()

st.sidebar.header("💰 Pricing Simulator")

price_median = data[price_col].median()

competitor_median = data[
    competitor_col
].median()

inventory_median = data[
    inventory_col
].median()

demand_median = data[
    sales_col
].median()

if pd.isna(price_median):

    price_median = 1000

if pd.isna(competitor_median):

    competitor_median = price_median

if pd.isna(inventory_median):

    inventory_median = 100

if pd.isna(demand_median):

    demand_median = 100

base_price = st.sidebar.number_input(
    "Base Price",
    min_value=1.0,
    value=float(
        max(
            price_median,
            1
        )
    )
)

competitor_price = st.sidebar.number_input(
    "Competitor Price",
    min_value=1.0,
    value=float(
        max(
            competitor_median,
            1
        )
    )
)

inventory = st.sidebar.number_input(
    "Inventory",
    min_value=1,
    value=int(
        max(
            inventory_median,
            1
        )
    )
)

demand = st.sidebar.number_input(
    "Expected Demand",
    min_value=1,
    value=int(
        max(
            demand_median,
            1
        )
    )
)

discount = st.sidebar.slider(
    "Discount %",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)

maximum_demand = max(
    float(
        data[sales_col].max()
    ),
    1
)

maximum_inventory = max(
    float(
        data[inventory_col].max()
    ),
    1
)

demand_ratio = min(
    demand /
    maximum_demand,
    1
)

inventory_pressure = max(
    0,
    1 -
    inventory /
    maximum_inventory
)

competitor_factor = (
    competitor_price /
    max(
        base_price,
        1
    )
)

multiplier = (
    0.85
    + demand_ratio * 0.30
    + inventory_pressure * 0.20
    + (competitor_factor - 1) * 0.10
    - (discount / 100) * 0.10
)

multiplier = float(
    np.clip(
        multiplier,
        0.70,
        1.50
    )
)

recommended_price = (
    base_price *
    multiplier
)

estimated_revenue = (
    recommended_price *
    demand
)

if demand_ratio >= 0.70:

    demand_level = "HIGH"

elif demand_ratio >= 0.40:

    demand_level = "MEDIUM"

else:

    demand_level = "LOW"

st.subheader(
    "💰 Dynamic Pricing Recommendation"
)

r1, r2, r3, r4 = st.columns(4)

r1.metric(
    "Demand Level",
    demand_level
)

r2.metric(
    "Price Multiplier",
    f"{multiplier:.2f}x"
)

r3.metric(
    "Recommended Price",
    f"₹{recommended_price:,.2f}"
)

r4.metric(
    "Estimated Revenue",
    f"₹{estimated_revenue:,.0f}"
)

st.subheader(
    "📈 Price Comparison"
)

price_chart = pd.DataFrame(
    {
        "Price": [
            base_price,
            competitor_price,
            recommended_price
        ]
    },
    index=[
        "Base Price",
        "Competitor Price",
        "Recommended Price"
    ]
)

st.bar_chart(
    price_chart
)

st.subheader(
    "📊 Demand Analysis"
)

demand_chart = data[
    [sales_col]
].head(100)

st.line_chart(
    demand_chart
)

st.subheader(
    "📦 Inventory Analysis"
)

inventory_chart = data[
    [inventory_col]
].head(100)

st.area_chart(
    inventory_chart
)

st.subheader(
    "💡 Business Recommendation"
)

if demand_level == "HIGH":

    st.success(
        "High demand detected. The system recommends "
        "increasing the selling price within a competitive "
        "range to improve revenue."
    )

elif demand_level == "MEDIUM":

    st.info(
        "Moderate demand detected. Maintain a balanced "
        "price close to the recommended value."
    )

else:

    st.warning(
        "Low demand detected. Consider reducing the price "
        "or increasing promotional activity."
    )

if model is not None:

    st.subheader(
        "🧠 Feature Importance"
    )

    importance = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_
        }
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )

result = data.copy()

result["Recommended_Price"] = (
    recommended_price
)

result["Price_Multiplier"] = (
    multiplier
)

result["Demand_Level"] = (
    demand_level
)

result["Estimated_Revenue"] = (
    estimated_revenue
)

csv_output = result.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Pricing Results",
    data=csv_output,
    file_name="dynamic_pricing_results.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Dynamic Pricing Engine | Machine Learning + Demand Analytics"
)