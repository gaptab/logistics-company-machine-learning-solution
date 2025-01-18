# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate dummy data for the transportation and logistics scenario
def generate_dummy_data():
    # Carrier pricing data
    data_carrier_pricing = pd.DataFrame({
        "lane_id": np.random.randint(1, 100, 500),
        "carrier_id": np.random.randint(1, 50, 500),
        "base_price": np.random.uniform(100, 1000, 500),
        "market_demand": np.random.uniform(0.5, 2.0, 500),  # market multiplier
        "covid_effect": np.random.uniform(0.8, 1.2, 500),  # COVID impact
    })
    # Calculate final price based on other features
    data_carrier_pricing["final_price"] = (
        data_carrier_pricing["base_price"]
        * data_carrier_pricing["market_demand"]
        * data_carrier_pricing["covid_effect"]
    )

    # Customer pricing data
    data_customer_pricing = pd.DataFrame({
        "customer_id": np.random.randint(1, 200, 1000),
        "cost": np.random.uniform(50, 500, 1000),
        "competitor_margin": np.random.uniform(5, 20, 1000),
        "recommended_margin": np.random.uniform(5, 25, 1000),  # Target variable
    })

    # Carrier recommendation data
    data_carrier_recommendation = pd.DataFrame({
        "lane_id": np.random.randint(1, 100, 1000),
        "carrier_id": np.random.randint(1, 50, 1000),
        "postings": np.random.randint(1, 20, 1000),
        "searches": np.random.randint(1, 15, 1000),
        "past_behavior_score": np.random.uniform(0, 1, 1000),
        "relevance_score": np.random.uniform(0, 1, 1000),  # Target variable
    })

    return data_carrier_pricing, data_customer_pricing, data_carrier_recommendation

# Generate dummy datasets
carrier_pricing, customer_pricing, carrier_recommendation = generate_dummy_data()

# Carrier Pricing Model
def carrier_pricing_model(data):
    print("\nTraining Carrier Pricing Model...")
    X = data[["base_price", "market_demand", "covid_effect"]]
    y = data["final_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Carrier Pricing Model R^2 Score: {score:.2f}")
    return model

# Customer Pricing Model
def customer_pricing_model(data):
    print("\nTraining Customer Pricing Model...")
    X = data[["cost", "competitor_margin"]]
    y = data["recommended_margin"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Customer Pricing Model R^2 Score: {score:.2f}")
    return model

# Carrier Recommendation Model
def carrier_recommendation_model(data):
    print("\nTraining Carrier Recommendation Model...")
    X = data[["postings", "searches", "past_behavior_score"]]
    y = data["relevance_score"]

    # Clustering carriers to identify patterns
    kmeans = KMeans(n_clusters=5, random_state=42)
    data["cluster"] = kmeans.fit_predict(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Carrier Recommendation Model R^2 Score: {score:.2f}")
    return model

# Visualize Carrier Recommendation Clusters
def plot_clusters(data):
    sns.scatterplot(x=data["postings"], y=data["searches"], hue=data["cluster"], palette="tab10")
    plt.title("Carrier Clusters Based on Postings and Searches")
    plt.xlabel("Postings")
    plt.ylabel("Searches")
    plt.show()

# Run all models
carrier_model = carrier_pricing_model(carrier_pricing)
customer_model = customer_pricing_model(customer_pricing)
recommendation_model = carrier_recommendation_model(carrier_recommendation)

# Plot clusters for carrier recommendation
plot_clusters(carrier_recommendation)

def save_dataframes_to_csv():
    # Generate dummy data
    data_carrier_pricing, data_customer_pricing, data_carrier_recommendation = generate_dummy_data()

    # Save each DataFrame to a CSV file
    data_carrier_pricing.to_csv("carrier_pricing.csv", index=False)
    data_customer_pricing.to_csv("customer_pricing.csv", index=False)
    data_carrier_recommendation.to_csv("carrier_recommendation.csv", index=False)

    print("DataFrames have been saved to CSV files successfully.")

# Call the function to save the data
save_dataframes_to_csv()
