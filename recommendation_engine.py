import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



# 1. Load Dataset


data = pd.read_csv("data/retail_data.csv")

print("Dataset loaded successfully!")
print(data.head())



# 2. Create User-Product Matrix


user_item_matrix = data.pivot_table(
    index="Customer ID",
    columns="Product Name",
    values="Rating",
    aggfunc="mean"
)

print("\nUser-Product Matrix:")
print(user_item_matrix)



# 3. Fill Missing Values


user_item_matrix_filled = user_item_matrix.fillna(0)

print("\nFilled User-Product Matrix:")
print(user_item_matrix_filled.head())



# 4. User-Based Collaborative Filtering


user_similarity = cosine_similarity(
    user_item_matrix_filled
)

print("\nUser Similarity Matrix:")
print(user_similarity)


def find_similar_customers(customer_id, n=5):

    if customer_id not in user_item_matrix_filled.index:
        print("Customer ID not found!")
        return []

    customer_index = user_item_matrix_filled.index.get_loc(
        customer_id
    )

    similarities = user_similarity[customer_index]

    similar_indices = similarities.argsort()[::-1][1:n + 1]

    similar_customers = user_item_matrix_filled.index[
        similar_indices
    ]

    return similar_customers



# 5. Find Similar Customers


customer_id = "CUST1018"

similar_customers = find_similar_customers(
    customer_id,
    n=5
)

print("\nSimilar Customers:")
print(similar_customers)



# 6. Item-Based Collaborative Filtering


item_user_matrix = user_item_matrix_filled.T

print("\nItem-User Matrix:")
print(item_user_matrix)


# Calculate similarity between products

item_similarity = cosine_similarity(
    item_user_matrix
)

print("\nItem Similarity Matrix:")
print(item_similarity)



# 7. Product Recommendation Function


def recommend_products(customer_id, n=5):

    if customer_id not in user_item_matrix_filled.index:
        print("Customer ID not found!")
        return []

    # Get customer's ratings
    customer_ratings = user_item_matrix_filled.loc[
        customer_id
    ]

    # Products already rated/purchased
    purchased_items = customer_ratings[
        customer_ratings > 0
    ].index.tolist()

    scores = {}

    
    # Item-Based Recommendation
    

    for item in purchased_items:

        item_index = item_user_matrix.index.get_loc(item)

        similarities = item_similarity[item_index]

        for i, similarity_score in enumerate(similarities):

            recommended_item = item_user_matrix.index[i]

            # Do not recommend already purchased products
            if recommended_item not in purchased_items:

                if similarity_score > 0:

                    scores[recommended_item] = (
                        scores.get(recommended_item, 0)
                        + similarity_score
                    )

    
    # Fallback Recommendation
    
    # If no similar products are found,
    # recommend products with highest average rating.

    if not scores:

        product_popularity = data.groupby(
            "Product Name"
        )["Rating"].mean()

        for product, rating in product_popularity.items():

            if product not in purchased_items:

                scores[product] = rating

    
    # Sort Recommendations
    

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:n]



# 8. Generate Recommendations


recommendations = recommend_products(
    customer_id,
    n=5
)



# 9. Display Final Recommendations


print("\n================================")
print("FINAL PRODUCT RECOMMENDATIONS")
print("================================")

print("Customer:", customer_id)

print("\nRecommended Products:")

if not recommendations:

    print("No recommendations available.")

else:

    for product, score in recommendations:

        print(
            f"- {product} (Score: {score:.3f})"
        )



# 10. Customer Purchase Counts


print("\nCustomer purchase counts:")

print(
    data["Customer ID"]
    .value_counts()
    .head(10)
)