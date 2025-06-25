Credit Scoring Business Understanding

1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord emphasizes quantitative risk measurement to ensure financial institutions manage credit risk responsibly. 
It mandates that models used in credit scoring must be transparent, auditable, and explainable—especially when determining capital reserves. 
This requirement influences our need to use interpretable models (such as Logistic Regression or Scorecards using Weight of Evidence) and 
to thoroughly document model design, assumptions, data sources, feature transformations, and risk thresholds. 
Regulators must be able to reproduce and justify credit decisions made by these models, making clarity as important as accuracy.

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of 
making predictions based on this proxy?

In the absence of an explicit “default” label, we must engineer a proxy target variable that 
approximates a user’s risk of default—e.g., based on behavioral signals like RFM (Recency, Frequency, Monetary) patterns, 
failed repayments, or transaction anomalies. While this enables us to train a model, proxies can introduce bias, 
especially if they imperfectly capture real-world defaults. Poorly chosen proxies may misclassify good customers as high risk,
resulting in lost revenue opportunities, or misidentify bad customers as low risk, causing increased credit losses. 
Therefore, validating the proxy through domain expertise and continuous feedback is essential.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, 
high-performance model (like Gradient Boosting) in a regulated financial context?

Criteria	             Simple Model (Logistic + WoE)	                       Complex Model (e.g., Gradient Boosting)
   
Interpretability	     High (easy to explain to regulators)	                Low (black-box behavior)
Regulatory Approval	     Easier to approve and audit	                        Harder to justify; requires model explanation tools
Performance	             May underperform on non-linear patterns	                Higher predictive power with rich feature sets
Deployment & Maintenance     Easier to maintain and update	                        Requires more compute and monitoring
Use Case Fit	             Preferred in high-stakes, regulated decisions	        Better in exploratory or high-volume environments

In highly regulated environments like banking, the balance often tilts toward interpretability, especially when decisions impact customers' 
financial access. However, hybrid approaches (e.g., using boosting with explainability tools like SHAP) can offer a middle ground.

