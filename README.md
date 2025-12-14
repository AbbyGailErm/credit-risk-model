## Credit Scoring Business Understanding
### Basel II Influence
The Basel II Accord emphasizes accurate measurement and management of credit risk. Banks must quantify the likelihood of borrower defaults to determine capital reserves. Therefore, our credit scoring model must be interpretable, transparent, and well-documented to satisfy regulatory requirements and enable auditability.
### Proxy Variable Necessity
Our dataset does not include a direct "default" label, which makes supervised learning challenging. Therefore, we create a proxy variable to categorize customers as high-risk or low-risk based on their transactional behavior. While this approach enables model training, it carries business risks such as potential misclassification of customers, which could result in approving loans for risky customers or denying loans to creditworthy customers. Continuous validation and monitoring of the proxy's predictive power are essential.
### Model Trade-offs
There is a trade-off between using simple, interpretable models and complex, high-performance models:
- **Simple Models (e.g., Logistic Regression with WoE):** Transparent, easy to explain, regulator-friendly, but may have lower predictive accuracy.
- **Complex Models (e.g., Gradient Boosting):** Can capture non-linear relationships and achieve higher accuracy, but are less interpretable and harder to justify in a regulated environment.

In a regulated financial context, maintaining model interpretability is crucial, though some performance gains from complex models may justify their use if transparency can be maintained.
