# logistics-company-machine-learning-solution
Machine learning solutions tailored for a transportation and logistics company

![alt text](https://github.com/gaptab/logistics-company-machine-learning-solution/blob/main/carrier_cluster.png)


Generate Dummy Data:

carrier_pricing: Models pricing based on base price, market demand, and COVID effect.

customer_pricing: Simulates cost, competitor margin, and recommended margin.

carrier_recommendation: Uses features like postings, searches, and past behavior scores.


Carrier Pricing Model: Linear regression predicts the final price based on base_price, market_demand, and covid_effect.

Customer Pricing Model: Random Forest Regressor predicts the recommended margin using cost and competitor_margin.

Carrier Recommendation Model: KMeans clustering segments carriers. A Random Forest Regressor predicts relevance scores for carrier recommendations.

Visualization: A scatter plot visualizes clusters in the carrier recommendation model.

How to Use: 
Replace the dummy data generation with actual company data.
Use the trained models to make predictions for real-world inputs.
Modify and fine-tune hyperparameters as needed for better accuracy.

459
