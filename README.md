# Spaceship Titanic Problem with Kaggle
# ADD IN REFERENCES WHERE I HAVE TALKED ABOUT RECOMMENDED PARAMETERS

## Executive Summary

This project explores the Spaceship Titanic problem using machine learning (classifiers), to accurately predict a target variable (whether a passenger would be transported to another dimension) depending on a set of given features. The potential benefits are adding to the data science research community by showcasing new ways of tacking this classification task, which will also help companies discover new ways to solve business problems.
The Spaceship Titanic problem is a new ML classification problem with the goal of getting an highly accurate predictive model to help push the boundaries of data science.

## Data Preprocessing

The dataset used for this project was sourced from Kaggle's ["Spaceship Titanic Data"](https://www.kaggle.com/c/spaceship-titanic/data). 
This dataset was produced by Kaggle and it contains various features like passenger expenditures on different services all the way to boolean features like did the customer have VIP, CyroSleep and so on. There are around 9,000 records within the train set, and with 13 features, there was enough data here to faciliate testing whether a classifier could accurately predict the target variable.

### Loading and Initial Exploration

I began by importing all the relevant python packages and then importing the train dataset which was in a csv format. The dataset was loaded into a Pandas DataFrame using the `pd.read_csv()` function. 

The initial exploration of the dataset was then conducted to gain insights into its structure and contents.

![Screenshot: DataFrame Info](screenshots/01.png)

### Data Selection and Renaming

From first thoughts, I planned to drop both 'PassengerId' and 'Name', to focus on the relevant columns for the problem. However doing further digging it seems that the passengers were split into groups and therefore share the same suffix e.g. '0001' depending on what group they were in. This would be a valuable categorical feature.

`df.drop(['Name'], axis=1, inplace=True)`

From here, I decided to split the PassengerId and Cabin features to capture their informations more clearly. 

`df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)`

`df.drop(['Cabin'], axis=1, inplace=True)`

`df[['PassengerGroup', 'PassengerNumber']] = df['PassengerId'].str.split('_', expand=True)`

`df.drop(['PassengerId'], axis=1, inplace=True)`

### Handling Missing Values

Rows containing missing values in the any of the records needed to be dropped as a large majority of machine learning models do not allow for unknown values.

I first tackled the room number feature, 'Num', and as it only had 199 missing values I just decided to these records from the dataset. This was because capturing these missing values with encoding techniques would confuse the final ML model, due to the high co-linearity.

`df.dropna(subset=['Num'], inplace=True)`

To carry on with the handling of the categorical feautres, I then created boolean columns (with int64 datatypes) to capture missing values for each categorical feature which had missing values.

`df['CryoSleep_Missing'] = df['CryoSleep'].isnull().astype(int) # int64 by default`

`df['Destination_Missing'] = df['Destination'].isnull().astype(int)`

`df['VIP_Missing'] = df['VIP'].isnull().astype(int)`

`df['HomePlanet_Missing'] = df['HomePlanet'].isnull().astype(int)`

After this, I one-hot encoded the categorical features so that ML models could interpret the categorical information. Note that this transformation is an requirement for a lot of ML models.

`df = pd.get_dummies(df, columns=['HomePlanet', 'Side', 'CryoSleep', 'Destination', 'VIP'], dtype=int)`

Then for the last categorical feature I converted the category labels directly into integers to highlight the ordinal aspect of this feature. In other words, I wanted to clearly show the ML model that there was an order to the decks of the ship i.e. Deck A is 1, Deck B is 2, and so on.

`deck_mapping = {
    'A': 7, # reverse order to highlight Deck A is highest deck
    'B': 6,
    'C': 5,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 1,
    'T': 0
}`

`df['Deck'] = df['Deck'].map(deck_mapping)`

After these transformations I was left with a dataframe looking like the following (this is just an subset due to the size of the dataframe):

![Screenshot: Missing Values 01](screenshots/02.png)

I now needed to handle the missing/invalid values within the numerical data. From some exploratory data anaylsis it was found that the 'Age' field had entries of 0. It doesn't make sense for someone to be 0 years old, therefore these values were set to missing with `df['Age'] = df['Age'].replace(0, np.nan)`.

These were the missing values counts for the numeric fields `df.isnull().sum()`:

Age	175
RoomService	177
FoodCourt	178
ShoppingMall	206
Spa	181
VRDeck	184

As there were lots of records in this dataset, I decided to impute the missing values for this features using a k-nearst neighbour (KNN) algorithm. This algorithm works by clustering data points around a set number of centroids (centre points), until the data points no longer change between clusters or a set maximum number of iterations is reached. I used the KNNImputer class from the sklearn package, and set it to have 5 clusters. To note, for medium sized datasets 4-6 is the advised number of clusters for this algorithm.

`imputer = KNNImputer(n_neighbors=5)`

`numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']`

`df[numeric_cols] = imputer.fit_transform(df[numeric_cols])`

And as you can see all missing values were removed from the numeric fields:

![Screenshot: Missing Values 02](screenshots/03.png)

### Standardizing the Features 

Standardizing features so that they have mean's of 0 and unit variances of 1, helps machine learning models converge (stabilize) during training alot quicker. In other words, it helps these models get to an optimal accuracy faster. This is due to the fact that a lot of weights and biases within Ml models are set to 0 in their initializations, and if features are standardized then the optimization algorithm (usually gradient descent) can reach the global minima quicker.

The equation for standardizing a feature is as follows:

![Screenshot: Standardization Equation](screenshots/04.png)

This equation can be done with numpy (a package within python) and setting our dataframe to an numpy array, however sci-kit learn provides methods to standardize features a lot more effiecently. To note it does not make sense to standardize categorical feautres which have been encoded, therefore only the numeric features were standardized.

`scaler = StandardScaler() # part of sci-kit learn library`

`numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']`

`df[numeric_cols] = scaler.fit_transform(df[numeric_cols])`

As you can see this standardized the numeric feautres:

![Screenshot: Standardization Equation](screenshots/05.png)

### Evaulating Feature Importance

As I now had I large number of features, 26 to be exact, I wanted to find out which features had the most importance for this classification problem. This was largely due to the fact that a large number of the features were encoded features, and therefore I suspected alot of co-linearilty as apparent. Co-linearilty is anything bad, but it just introduces rendunant features, i.e. you could get the same information/worth out of one feature instead of two.

I split the features and target varaible:

`X_train = df.drop(['Transported'], axis=1)`

`y_train = df['Transported']`

And then used the Random Forest classification model, from the sklearn package, to understand which features had the most importance. This works because as the Random Forest is creating it's decision trees for the task, it also tracks the importance of each variable (feature). This happens because the model uses these values to understand which features increase the model accuracy most by having greater impurity reductions.

`from sklearn.ensemble import RandomForestClassifier`

`feat_labels = df.columns.drop('Transported')`

`forest = RandomForestClassifier(n_estimators=500, random_state=1)`

`forest.fit(X_train, y_train)`

`importances = forest.feature_importances_`

`indices = np.argsort(importances)[::-1]`

`for f in range(X_train.shape[1]):`

     `print("%2d) %-*s %f" % (f + 1, 30,`
     
                             `feat_labels[indices[f]],`
                             
                             `importances[indices[f]]))`
                             
`plt.title('Feature importance')`

`plt.bar(range(X_train.shape[1]),`

         `importances[indices],`
         
         `align='center')`
         
`plt.xticks(range(X_train.shape[1]),`

            `feat_labels[indices], rotation=90)`
            
`plt.xlim([-1, X_train.shape[1]])`

`plt.tight_layout()`

![Screenshot: Feature Importance Graph](screenshots/06.png)

It can be seen that in the case of training this Random Forest Classifier, the features 'Num' (room_number), 'PassengerGroup' and 'Age', are the three features that have the highest importance. I was surprised by these results as I thought the 'Deck' feature would have more of an importance. This graph also highlights the number of features which have very little importance or value to the model. I will discuss these more in the next chapter.

### Reducing Model Dimensionality for Visualisation Purposes

Currently there are 26 features, a mix of categorical and numerical, within the dataset. To visualise the spread of the classes (0, 1) within 'Transported' (the target variable), I would like to reduce the dimensions to 2, whilst keeping the majority of information. This is so that I can then plot the Transported variable on a 2-dimensional scatter plot. This will then allow me to observe whether the classes are linearly seperable within Transported.

To reduce the number of dimensions, a common technique is using Principal Component Analysis. In short, PCA creates a covariance matrix, obtains the eigenvectors and eigenvalues from this matrix, construct a projection matrix (W) from the (sorted) top k eigenvectors, and lastly transform the orginal matrix (your features) using the project matrix (W) to get the new k-dimensional subspace.

The below is an graph which highlights one of the processes within PCA. It's shows that most of the variance within all of the k-features can be explained by just two featues.

![Screenshot: PCA](screenshots/07.png)

And this is the whole idea behind PCA, trying to retain the most amount of information whilst decreasing the dimensional space to k. k here, is a number you decide.

Anyway, I used the PCA class from sklearn to efficiently reduce my 26 dimensional feature dataset, down to 2 dimensions (or 2 features).

![Screenshot: PCA](screenshots/08.png)

With the features now down to two dimensions, I could visualise the Transported field:

![Screenshot: PCA plot code](screenshots/09.png)

![Screenshot: PCA plot](screenshots/10.png)

From this plot we can now see that there is a definite pattern however the classes (yellow & purple), are mixed together with no clear linearl seperability at all. Therefore, my initial thoughts were that decision trees would be best for this classification task due to their flexible nature. I will talk more about this in the next chapter.

### Final Dataset

![Screenshot: Final Dataset](screenshots/01_initial_exploration/screenshot10.png)

The resulting cleaned and standardized dataset was saved for future use, and it contains 'id', 'weight_kg', and 'price' columns.

## Solving the Knapsack Problem

In this section, we walk through the steps of solving the Knapsack Problem using Google OR-Tools with the cleaned and standardized dataset. The Knapsack Problem involves selecting items to maximize value while staying within capacity constraints, which is a valuable optimization technique for retail organizations.

### Loading the Cleaned Dataset

We begin by loading the previously cleaned and standardized dataset, which includes 'id', 'weight_kg' and 'price' columns. 
This dataset is now ready for solving the Knapsack Problem.

![Screenshot: Loading the Cleaned Dataset](screenshots/02_knapsack_problem/screenshot1.png)

### Random Sampling for Reproducibility

For reproducibility, we set a random seed and shuffle the dataset. We then select a random subset of items to solve the Knapsack Problem.

```python
random_seed = 42
random.seed(random_seed)

shuffled_df = df.sample(frac=1, random_state=random_seed)

num_samples = 100
random_subset = shuffled_df.head(num_samples)
```
![Screenshot: Sampled Subset](screenshots/02_knapsack_problem/screenshot2.png)

### Knapsack Problem Solver

We initialize the Knapsack Solver with the desired solver type and set the capacity of the knapsack (maximum weight allowed). 
This solver type is suitable for multidimensional problems.

```python
solver = knapsack_solver.KnapsackSolver(
    knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
    "KnapsackExample",
)

# Capacity of the knapsack (maximum weight allowed)
capacity = 25000
```

### Preparing Data for the Solver

We prepare the data for the solver by converting weights and values to integers as required by OR-Tools. 
The number of items in the subset is also determined.

```python
weights = random_subset['weight_kg'].tolist()
values = random_subset['price'].tolist()

# Convert weights and values to integers (required by OR-Tools)
weights = [int(w * 1000) for w in weights]  # Convert to grams (integer)
values = [int(v * 100) for v in values]  # Convert to currency (integer)

# Number of items
num_items = len(random_subset)
```

### Solver Initialization and Solution

We initialize the solver with the prepared data and solve the Knapsack Problem to find the selected items (1 for selected, 0 for not selected).

```python
# Set the solver parameters
solver.init(values, [weights], [capacity])
solver.solve()

# Get the selected items (1 for selected, 0 for not selected)
selected_items = [solver.best_solution_contains(i) for i in range(num_items)]
```

### Results and Analysis

We present the output, and verify the results by summing the total price and weight. 
In this step, we encounter a discrepancy in the total weight, which prompts further investigation.

![Screenshot: Results and Analysis](screenshots/02_knapsack_problem/screenshot3.png)

![Screenshot: Results and Analysis Discrepancy](screenshots/02_knapsack_problem/screenshot4.png)

### Addressing Discrepancy

We address the discrepancy by converting the 'kg' weight column into grams. 
After performing the conversion and re-sampling, we solve the Knapsack Problem again.

```python
def convert_grams(weight_kg):
    weight_grams = math.ceil(weight_kg * 1000)

    try:
        return int(weight_grams)
    except ValueError:
        return None

df['weight_grams'] = df['weight_kg'].apply(convert_grams)
```

### Reanalysis of Results

We reanalyze the results with the converted weight in grams, including the total weight in kilograms and grams.
The discrepancy still exists, but the solver does get the total weight below the target of 25,000 grams or 25kg

![Screenshot: Reanalysis of Results](screenshots/02_knapsack_problem/screenshot5.png)

![Screenshot: Reanalysis of Results](screenshots/02_knapsack_problem/screenshot6.png)

## Recommendations for Future Iterations

1. **Multiple Knapsacks:** Future iterations of this project could expand from solving a single knapsack problem to solving multiple knapsacks simultaneously. This would be valuable for scenarios where there are multiple constraints or resources to consider.

2. **Complex Packing Problems:** Consider tackling more complex packing problems, such as 2D or 3D bin packing, which have practical applications in logistics, manufacturing, and resource allocation.

## Challenges Encountered

One challenge encountered during the project was dealing with floating-point arithmetic limitations, which can affect the precision of calculations when working with values like weights and prices.
While the discrepancy is small enough for this project, it's essential to be aware of such limitations in real-world applications.

## References

1. [Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020)
2. [Google OR-Tools Documentation](https://developers.google.com/optimization/pack/knapsack)
3. Medium Articles:
   - [Exploring the Bin Packing Problem](https://medium.com/swlh/exploring-the-bin-packing-problem-f54a93ebdbe5)
   - [The Knapsack Problem: A Guide to Solving One of the Classic Problems in Computer Science](https://levelup.gitconnected.com/the-knapsack-problem-a-guide-to-solving-one-of-the-classic-problems-in-computer-science-7b63b0851c89)
