# K-Means Clustering — Mall Customer Segmentation

An Unsupervised Learning project that groups mall customers into natural segments based on their **Annual Income** and **Spending Score** — without using any target/label.

## 📁 Files

| File | Purpose |
|---|---|
| `KMeans_Clustering_Mall_Customers.ipynb` | Complete Jupyter notebook — from data understanding to final cluster interpretation |

## 📊 Dataset

**Mall Customer Segmentation Dataset** (Kaggle: `vjchoudhary7/customer-segmentation-tutorial-in-python`)

The dataset is loaded directly from a public raw-CSV mirror on GitHub, so no manual download is needed:
```
https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv
```

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Customer's age |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Spending behavior score assigned by the mall |

**Total Rows:** 200 customers

## 🔧 Pipeline — What's Inside the Notebook

1. **Data Understanding** — shape, dtypes, missing value checks
2. **EDA** — distributions of Age/Income/Spending Score, plus an Income vs. Spending scatter plot that visually reveals natural groupings
3. **Feature Selection & Scaling** — selected `Annual_Income` and `Spending_Score`, applied `StandardScaler` (essential since K-Means is a distance-based algorithm)
4. **Finding the Best K:**
   - **Elbow Method** — plotted WCSS (Inertia) for K = 1 to 10
   - **Silhouette Score** — a second method used to confirm the Elbow Method's result
5. **Final Model Training** — trained `KMeans(n_clusters=5)`
6. **Visualization** — color-coded scatter plot with cluster centroids
7. **Business Interpretation** — assigned each cluster a real-world, business-friendly name (e.g. "Premium Customers", "Impulsive Spenders")

## 🏆 Results

- **Best K = 5** (confirmed by both the Elbow Method and Silhouette Score)
- **5 Customer Segments identified:**

| Cluster | Name | Annual Income | Spending Score |
|---|---|---|---|
| 0 | Average Customers | Medium (~$55k) | Medium (~50) |
| 1 | Premium Customers | High (~$87k) | High (~82) |
| 2 | Impulsive Spenders | Low (~$26k) | High (~79) |
| 3 | Cautious Rich | High (~$88k) | Low (~17) |
| 4 | Careful Customers | Low (~$26k) | Low (~21) |

## 💡 Business Insights

- **Premium Customers** (High Income, High Spending) → target with loyalty programs / premium offers
- **Impulsive Spenders** (Low Income, High Spending) → attract with discounts / installment plans
- **Cautious Rich** (High Income, Low Spending) → try to convert with personalized recommendations
- **Average / Careful Customers** → general marketing is sufficient

## 🚀 How to Run the Notebook

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook KMeans_Clustering_Mall_Customers.ipynb
```

Then select **Run All Cells** — the dataset loads automatically from GitHub, no extra files needed.

## 🛠️ Tech Stack

- Python, pandas, numpy
- matplotlib, seaborn (visualization)
- scikit-learn (`KMeans`, `StandardScaler`, `silhouette_score`)

## 📌 Key Concepts Covered

- K-Means Algorithm (centroid-based clustering)
- Euclidean Distance & WCSS (Inertia)
- K-Means++ Initialization
- Elbow Method
- Silhouette Score
- Feature Scaling (why it matters for distance-based algorithms)
