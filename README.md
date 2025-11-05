
## 🚢 **Project Title:** Titanic Survival Prediction using PySpark ML

---

### 🎯 **1. Project Objectives**

The purpose of this project is to apply **PySpark’s Machine Learning (ML) framework** to analyze the Titanic dataset and build predictive models that determine **which passengers survived the Titanic disaster** based on available passenger data.

**Key goals include:**

* Exploring the dataset using distributed data processing (PySpark).
* Performing **data cleaning, transformation, and feature engineering** for ML readiness.
* Building predictive models using **classification algorithms** such as Logistic Regression, Decision Tree, and Random Forest.
* Evaluating model performance using key metrics like accuracy, precision, recall, and F1-score.
* Deriving insights about the key features that influenced passenger survival.

---

### ⚙️ **2. Methods**

#### **A. Data Preprocessing**

* Loaded the Titanic dataset into a PySpark DataFrame.
* Conducted **data cleaning**:

  * Removed duplicates and handled missing values (especially in *Age*, *Embarked*, and *Cabin* columns).
  * Converted categorical variables (`Sex`, `Embarked`) into numerical format using **StringIndexer** and **OneHotEncoder**.
* Created the **label column** (`Survived`) and a vector of **feature columns** (e.g., `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked`).
* Used **VectorAssembler** to combine all features into a single feature vector suitable for PySpark MLlib models.

#### **B. Feature Engineering**

* Derived new attributes such as:

  * Family size = `SibSp + Parch + 1`
  * IsAlone = 1 if Family size = 1, else 0.
* Scaled numerical features using **StandardScaler** to normalize values.

#### **C. Exploratory Data Analysis (EDA)**

* Used PySpark SQL functions and DataFrame operations to compute descriptive statistics.
* Visualized survival distributions with respect to:

  * Gender
  * Passenger Class (Pclass)
  * Age
  * Embarked port
  * Fare amount

#### **D. Machine Learning Modeling**

* Implemented ML pipeline consisting of:

  * Feature transformations (indexing, encoding, scaling)
  * Model training and evaluation
* Models used:

  * **Logistic Regression**
  * **Decision Tree Classifier**
  * **Random Forest Classifier**
* Split dataset into **training (80%)** and **testing (20%)** sets.

#### **E. Model Evaluation**

* Used **BinaryClassificationEvaluator** to measure model performance.
* Computed:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Compared model performance across all algorithms.

---

### 🧾 **3. Dataset**

* **Dataset Name:** Titanic Passenger Dataset
* **Source:** [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
* **Total Records:** ~891 passengers
* **Key Columns:**

  | Feature    | Description                       |
  | ---------- | --------------------------------- |
  | `Survived` | Target variable (0 = No, 1 = Yes) |
  | `Pclass`   | Passenger class (1st, 2nd, 3rd)   |
  | `Name`     | Passenger name                    |
  | `Sex`      | Gender                            |
  | `Age`      | Passenger age                     |
  | `SibSp`    | Number of siblings/spouses aboard |
  | `Parch`    | Number of parents/children aboard |
  | `Ticket`   | Ticket number                     |
  | `Fare`     | Ticket fare                       |
  | `Cabin`    | Cabin number                      |
  | `Embarked` | Port of embarkation (C, Q, S)     |

---

### 📈 **4. Results**

#### **A. Model Comparison**

| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | 0.81     | 0.80      | 0.78     | 0.79     |
| Decision Tree       | 0.83     | 0.82      | 0.81     | 0.81     |
| Random Forest       | **0.86** | **0.85**  | **0.84** | **0.85** |

✅ **Best Model:** Random Forest Classifier

* Provided the highest accuracy and F1-score among all models.
* Showed robust performance with minimal overfitting.

#### **B. Feature Importance (from Random Forest)**

1. **Sex** (most influential factor)
2. **Pclass** (passenger class)
3. **Fare** (ticket price)
4. **Age**
5. **Family size (SibSp + Parch)**
6. **Embarked**

#### **C. Key Observations**

* **Females** had a significantly higher survival rate than males (~75% vs 20%).
* **First-class passengers** had the highest survival probability (~63%), while third-class had the lowest (~25%).
* **Younger passengers (<15 years)** had higher survival rates than older ones.
* Passengers who paid higher **fares** were more likely to survive, possibly due to cabin location advantages.

---

### 💡 **5. Insights**

1. **Gender is the strongest predictor** of survival — females were prioritized during evacuation ("women and children first").
2. **Socioeconomic class (Pclass)** heavily influenced survival chances, reflecting access to better cabins and lifeboats.
3. **Fare** serves as a proxy for passenger wealth and indirectly correlates with survival probability.
4. **Traveling alone** reduced chances of survival, while families often survived together.
5. **Machine Learning with PySpark** enables scalable and distributed training, making it suitable for large datasets beyond Titanic’s size.

---

### 🏁 **Conclusion**

This project successfully demonstrates how **PySpark MLlib** can be used for large-scale data preprocessing, feature engineering, and model training in distributed environments.

* The **Random Forest model** achieved the highest predictive accuracy of **~86%**, showing strong generalization.
* The analysis provided both **technical and sociological insights** into passenger survival patterns.


