## 📘 **Analyzing Data with Pandas and Visualizing Results with Matplotlib**

### 👨‍🎓 Assignment Objectives:
- Load and explore a dataset using `pandas`.
- Perform basic analysis.
- Visualize the data using `matplotlib` (and optionally `seaborn`).
- Present key insights clearly.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
%matplotlib inline

### 📂 **Task 1: Load and Explore the Dataset**

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({i: species for i, species in enumerate(iris.target_names)})
df.head()

df.info()
df.isnull().sum()
df.describe()

### 🧹 **Clean the Dataset**

df_cleaned = df.dropna()

### 📊 **Task 2: Basic Data Analysis**

df.describe()

df.groupby('species').mean()

### 📈 **Task 3: Data Visualization**

plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['petal length (cm)'], label=species)
plt.title("Petal Length over Index by Species")
plt.xlabel("Index")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

avg_sepal_width = df.groupby('species')['sepal width (cm)'].mean()
avg_sepal_width.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title("Average Sepal Width per Species")
plt.ylabel("Sepal Width (cm)")
plt.xlabel("Species")
plt.grid(axis='y')
plt.show()

plt.hist(df['petal length (cm)'], bins=20, color='purple', edgecolor='black')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

### 🧠 **Findings / Observations**
- **Setosa** species stands out clearly with shorter petals and slightly wider sepals.
- **Virginica** has the longest petals, while **Versicolor** is intermediate.
- Positive correlation between sepal length and petal length.
- Visualizations help easily distinguish between the species based on feature dimensions.

### 💡 **Error Handling Example**

try:
    df_check = pd.read_csv("iris.csv")
except FileNotFoundError:
    print("File not found. Using sklearn dataset instead.")

### 📝 Submission Checklist
- ✅ Dataset loaded and cleaned
- ✅ Basic analysis and grouping done
- ✅ 4 types of visualizations included
- ✅ Plots are labeled with titles and legends
- ✅ Key insights mentioned