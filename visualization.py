import pandas as pd
import matplotlib.pyplot as plt



# 1. Load Dataset


data = pd.read_csv("data/retail_data.csv")

print("Dataset loaded successfully!")
print(data.head())



# 2. Product Category Sales


category_sales = data.groupby(
    "Product Category"
)["Price"].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

category_sales.plot(kind="bar")

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



# 3. Product Popularity


product_count = data["Product Name"].value_counts()

plt.figure(figsize=(10, 6))

product_count.plot(kind="bar")

plt.title("Product Popularity")
plt.xlabel("Product Name")
plt.ylabel("Number of Purchases")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



# 4. Rating Distribution


plt.figure(figsize=(8, 5))

data["Rating"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.xticks(rotation=0)
plt.tight_layout()

plt.show()



# 5. Customer Purchase Count


customer_count = data["Customer ID"].value_counts().head(10)

plt.figure(figsize=(10, 6))

customer_count.plot(kind="bar")

plt.title("Top 10 Customers by Purchase Count")
plt.xlabel("Customer ID")
plt.ylabel("Number of Purchases")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



# 6. Average Rating by Product Category


category_rating = data.groupby(
    "Product Category"
)["Rating"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))

category_rating.plot(kind="bar")

plt.title("Average Rating by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Rating")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


print("\n================================")
print("VISUALIZATION COMPLETED")
print("================================")