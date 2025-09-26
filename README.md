# CS5710 Machine Learning - Homework 2

**University of Central Missouri**  
Department of Computer Science & Cybersecurity  
Fall 2025  

---

## Student Info
- **Name:** Ajay Muppa  
- **Course:** CS5710 Machine Learning  
- **Assignment:** Homework 2  

---

## Repository Contents
- `homework2_partA.pdf` → Solutions to Part A (calculation-based questions).  
- `homework2_partB.py` → Python code for Part B (Decision Tree, kNN, evaluation).  
- `confusion_matrix.png` → Heatmap of confusion matrix for kNN (k=5).  
- `roc_curve.png` → ROC curve plot with AUC values for kNN (k=5).  
- `knn_boundary_k1.png`, `knn_boundary_k3.png`, `knn_boundary_k5.png`, `knn_boundary_k10.png` → Decision boundary plots for kNN using sepal length & width.  

---

## How to Run
1. Install required dependencies:
   ```bash
   pip install scikit-learn matplotlib seaborn
2. Run the Part B script:
   ```bash
   python homework2_partB.py
3. The script will:
   Print Decision Tree accuracies (depth=1,2,3).
   Save kNN decision boundary plots for k=1,3,5,10.
   Print confusion matrix and classification report for k=5.
   Save confusion matrix heatmap and ROC curves.

---

## Results Interpretation

## Part A (Analytical)

- Decision stump error = 25% vs 0% for memorizer.

- All features had equal training error in splitting (16.7%).

- Best split (by information gain) = Exercise.

- Metrics show accuracy 80%, but precision/recall/F1 are more informative under imbalance.

- Cross-validation shows k=5 generalizes best.

## Part B (Programming)

- Decision Tree:

Depth=1 → underfits (low accuracy).

Depth=2 → better fit, balanced generalization.

Depth=3 → near-perfect accuracy, risk of overfitting.

- kNN with 2 features (decision boundaries):

k=1 → very jagged, high variance.

Larger k → smoother, more generalized boundaries.

- kNN with all features (k=5):

Confusion matrix shows almost perfect classification (1–2 errors max).

Accuracy ≈ 97–98%.

ROC curve AUC ≈ 0.99+, indicating excellent classifier performance.


---


## Notes

All plots are generated automatically and saved in the repo.

Code is fully commented and reproducible.



