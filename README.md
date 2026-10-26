Case Study 1: Online Retail Basket Clustering

The retail analysis explores 7,501 transactions containing 20 item slots each. After cleaning, the dataset contains 119 distinct products. The average basket has 3.91 products, the median is 3, and mineral water is the most frequently listed product, appearing in 1,788 baskets.

Transactions are represented as binary product-presence vectors. PCA, K-means, hierarchical clustering, DBSCAN, and spectral clustering are used to explore basket similarities. The first two principal components explain 11.18% of the variance. Clustering results show weak separation: K-means has a sampled silhouette of 0.1285, while hierarchical clustering and DBSCAN produce partitions that do not support valid scores on the evaluation subset. The results do not justify confident basket segmentation.

Case Study 4: Student Dropout and Academic Success

The student analysis uses 4,424 records with 36 predictors and three recorded outcomes: Dropout, Enrolled, and Graduate. It explores outcome balance, admission grades, first-semester performance, and age patterns.

Welch’s ANOVA found a difference in mean first-semester grades across the outcome groups: F(2, 1755.4) = 522.433, p = 9.61 × 10⁻¹⁷⁹, η² = 0.244. A stratified test set was used to compare logistic-regression models. The intake-only model achieved 0.563 accuracy and 0.532 macro F1. The model using information available after semester one achieved 0.706 accuracy and 0.666 macro F1. Enrolled was the most difficult class to identify.

These findings describe associations in this dataset. They do not establish causes or predict an individual student’s future with certainty.

Files

combined_case_studies_1_and_4.ipynb — notebook with code, outputs, charts, and explanations.

combined_case_studies_1_and_4.py — Python script for running both analyses.

Online retail_new.csv — retail transaction data.

Studentdata.csv — student outcome data.

Running the Notebook

In Google Colab, open the notebook and run the cells from top to bottom. Upload Online retail_new.csv when prompted for the retail analysis, then upload Studentdata.csv for the student analysis.

Running the Python Script Locally

Install the required packages:

pip install pandas numpy matplotlib seaborn scipy scikit-learn threadpoolctl

Run both analyses from the folder containing the CSV files:

python combined_case_studies_1_and_4.py --retail-csv "Online retail_new.csv" --student-csv "Studentdata.csv"

Charts are saved in the case_study_outputs folder by default.

Limitations

The retail clustering was performed on a reproducible sample, and the results show weak group separation. The student model was evaluated using a random stratified split from one historical cohort; future-cohort validation is still needed. The notebook does not run pairwise post-hoc tests for the ANOVA or a separate tuition-fee group comparison. Any student predictions should be used only to support voluntary, human-reviewed assistance—not to deny opportunities or services.
