# Spaceship Titanic Problem with Kaggle

![Screenshot: Picture of outer space](screenshots/space.png)

## Executive Summary

This project explores the Spaceship Titanic problem using machine learning (classifiers), to accurately predict a target variable (whether a passenger would be transported to another dimension) depending on a set of given features. The potential benefits are adding to the data science research community by showcasing new ways of tacking this classification task, which will also help companies discover new, more powerful ways to solve business problems with ML like identifying which customers are likely to churn.
The Spaceship Titanic problem is a new ML classification problem with the goal of getting an highly accurate predictive model to help push the boundaries of data science. I will end this section with the hypothesis which will be used as guiadance throughout this project: Passenger demographics, travel details, and onboard activities significantly influence the likelihood of being transported to an alternate dimension during the Spaceship Titanic’s collision with a spacetime anomaly.

## Ethics, bias, privacy, legal, regulatory obligatons

This is important as I need to make sure I am adhere to any legal obligations that may be apparent within this project. Going along the list starting with ethics, this should not be an issue due to all data being provided by Kaggle and it's documentation specificially stating this data has been provided for use to create an predictive machine learning model. Also to note, no other data is being pulled in to help increase the model's performance. Next, bias. This could be apparent due to the dataset size. Even with a dataset of around 10,000 records, the quality and diversity of these data records still may be poor and not reflective of the true population. Therefore, we can only assume that these are the only passengers in the incident and all data was recorded accurately. Lastly for privacy, legal and regulatory obligations, it's in an identicial position to the ethics for this project. This is because all data within this project has been provided by Kaggle with the appropiate licenses to state that this data is allowed to be used and made available publicly. 

## Data Preprocessing

The dataset used for this project was sourced from Kaggle's ["Spaceship Titanic Data"](https://www.kaggle.com/c/spaceship-titanic/data). 
This dataset was produced by Kaggle and it contains various features like passenger expenditures on different services all the way to boolean features like did the customer have VIP, CyroSleep and so on. There are around 9,000 records within the train set, and with 13 features, there was enough data here to faciliate testing whether a classifier could accurately predict the target variable.

### Loading and Initial Exploration

I began by importing all the relevant python packages and then importing the train dataset which was in a csv format. The dataset was loaded into a Pandas DataFrame using the `pd.read_csv()` function. 

The initial exploration of the dataset was then conducted to gain insights into its structure and contents.

![Screenshot: DataFrame Info](screenshots/01.png)

### Data Selection and Renaming

From first thoughts, I planned to drop both 'PassengerId' and 'Name', to focus on the relevant columns for the problem. However doing further digging it seems that the passengers were split into groups and therefore share the same preffix e.g. '0001' depending on what group they were in. This would be a valuable categorical feature.

From here, I decided to split the PassengerId and Cabin features to capture their informations more clearly. 

![Screenshot: DataFrame Info](screenshots/16.png)

### Handling Missing Values

Rows containing missing values in the any of the records needed to be dropped as a large majority of machine learning models do not allow for unknown values.

I first tackled the room number feature, 'Num', and as it only had 199 missing values I just decided to drop these records from the dataset. This was because capturing these missing values with encoding techniques would confuse the final ML model, due to the high co-linearity.

![Screenshot: DataFrame Info](screenshots/17.png)

To carry on with the handling of the categorical feautres, I then created boolean columns (with int64 datatypes) to capture missing values for each categorical feature which had missing values.

![Screenshot: DataFrame Info](screenshots/18.png)

After this, I one-hot encoded the categorical features so that ML models could interpret the categorical information. Note that this transformation is an requirement for a lot of ML models.

`df = pd.get_dummies(df, columns=['HomePlanet', 'Side', 'CryoSleep', 'Destination', 'VIP'], dtype=int)`

Then for the last categorical feature I converted the category labels directly into integers to highlight the ordinal aspect of this feature. In other words, I wanted to clearly show the ML model that there was an order to the decks of the ship i.e. Deck A is 1, Deck B is 2, and so on.

![Screenshot: Missing Values 02](screenshots/19.png)

After these transformations I was left with a dataframe looking like the following (this is just an subset due to the size of the dataframe):

![Screenshot: Missing Values 01](screenshots/02.png)

I now needed to handle the missing/invalid values within the numerical data. From some exploratory data anaylsis it was found that the 'Age' field had entries of 0. It doesn't make sense for someone to be 0 years old, therefore these values were set to missing with `df['Age'] = df['Age'].replace(0, np.nan)`.

These were the missing values counts for the numeric fields `df.isnull().sum()`:

![Screenshot: Missing Values 02](screenshots/20.png)

As there were lots of records in this dataset, I decided to impute the missing values for these features using a k-nearst neighbour (KNN) algorithm. This algorithm works by clustering data points around a set number of centroids (centre points), until the data points no longer change between clusters or a set maximum number of iterations is reached. I used the KNNImputer class from the sklearn package, and set it to have 5 clusters. To note, for medium sized datasets 4-6 is the advised number of clusters for this algorithm. (ADD REFERENCE/EVIDENCE HERE)

![Screenshot: Missing Values 02](screenshots/21.png)

And as you can see all missing values were removed from the numeric fields:

![Screenshot: Missing Values 02](screenshots/03.png)

### Standardizing the Features 

Standardizing features so that they have mean's of 0 and unit variances of 1, helps machine learning models converge (stabilize) during training alot quicker. In other words, it helps these models get to an optimal accuracy faster. This is due to the fact that a lot of weights and biases within Ml models are set to 0 in their initializations, and if features are standardized then the optimization algorithm (usually gradient descent) can reach the global minima quicker.

The equation for standardizing a feature is as follows:

![Screenshot: Standardization Equation](screenshots/04.png)

This equation can be done with numpy (a package within python) and setting our dataframe to an numpy array, however sci-kit learn provides methods to standardize features a lot more effiecently. To note it does not make sense to standardize categorical feautres which have been encoded, therefore only the numeric features were standardized.

![Screenshot: Missing Values 02](screenshots/22.png)

As you can see this standardized the numeric feautres:

![Screenshot: Standardization Equation](screenshots/05.png)

### Evaulating Feature Importance

As I now had I large number of features, 26 to be exact, I wanted to find out which features had the most importance for this classification problem. This was largely due to the fact that a large number of the features were encoded features, and therefore I suspected alot of co-linearilty was apparent. Co-linearilty isn't anything bad, but it just introduces rendunant features, i.e. you could get the same information/worth out of one feature instead of two.

I split the features and target varaible:

`X_train = df.drop(['Transported'], axis=1)`

`y_train = df['Transported']`

And then used the Random Forest classification model, from the sklearn package, to understand which features had the most importance. This works because as the Random Forest is creating it's decision trees for the task, it also tracks the importance of each variable (feature). This happens because the model uses these values to understand which features increase the model accuracy most by having greater impurity reductions.

![Screenshot: Feature Importance Code](screenshots/12.png)

![Screenshot: Feature Importance Graph](screenshots/06.png)

It can be seen that in the case of training this Random Forest Classifier, the features 'Num' (room_number), 'PassengerGroup' and 'Age', are the three features that have the highest importance. I was surprised by these results as I thought the 'Deck' feature would have more of an importance. This graph also highlights the number of features which have very little importance or value to the model. I will discuss these more in the next chapter.

### Reducing Model Dimensionality for Visualisation Purposes

Currently there are 26 features, a mix of categorical and numerical, within the dataset. To visualise the spread of the classes (0, 1) within 'Transported' (the target variable), I would like to reduce the dimensions to 2, whilst keeping the majority of information. This is so that I can then plot the Transported variable on a 2-dimensional scatter plot. This will then allow me to observe whether the classes are linearly seperable within Transported.

To reduce the number of dimensions, a common technique is using Principal Component Analysis. In short, PCA creates a covariance matrix, obtains the eigenvectors and eigenvalues from this matrix, constructs a projection matrix (W) from the (sorted) top k eigenvectors, and lastly transform the orginal matrix (your features) using the projection matrix (W) to get the new k-dimensional subspace.

The below is an graph which highlights one of the processes within PCA. It's shows that most of the variance within all of the k-features can be explained by just two featues.

![Screenshot: PCA](screenshots/07.png)

And this is the whole idea behind PCA, trying to retain the most amount of information whilst decreasing the dimensional space to k. k here, is a number you decide.

Anyway, I used the PCA class from sklearn to efficiently reduce my 26 dimensional feature dataset, down to 2 dimensions (or 2 features).

![Screenshot: PCA](screenshots/08.png)

With the features now down to two dimensions, I could visualise the Transported field:

![Screenshot: PCA plot code](screenshots/09.png)

![Screenshot: PCA plot](screenshots/10.png)

From this plot we can now see that there is a definite pattern however the classes (yellow & purple), are mixed together with no clear linear seperability at all. The next stage is now to decide which model to use and carry out training and testing.

## Training and testing a model

In this section, we walk through the steps of training a ML model to solve the classification problem, with the cleaned and standardized features. The classification problem is to accuractely predict which passengers will be transported to an alternate dimension, therefore the goal is to minimize the error function (the loss). 

### Deciding on what model to use

As our dataset contains a lot features that are both numeric and categorical I decided to start of my experiments by using decision trees with boosting. This would be an ensemble model, as multiple decision trees (weak learners) are brought together to create a model which is accurate (low bias) but also genaralizes well to unseen data (low variance). This is where boosting excels as each decision tree added to the model focuses on correcting the residuals (errors) from the previous decision tree. To prevent overfitting too, regularization would have be incorporated to prevent certain weights within the model becoming to 'strong', by adding in weight decay (L2 regularization). Or setting certain features' weights to 0 which effectively removes those irrevelant features, also known as (L1 regularization).
With all these requirements listed out it is clear that XGBoost would definetly be a good model to try first for this task. The XGBoost model has a gradient boosting algorithm which helps it learn the optimal weights effiecently. Here is a general outline of how the gradient boosting algorithm works for a binary classification problem of either 0 or 1.

![Screenshot: Gradient Boosting Outline](screenshots/11.png)

I came across the XGBoost model after doing resesrch into emerging ML models. This model was and still is one of the best models for obtaining highly accurate predictions, therefore it has seen a consistent upwards trend in it's number of use cases.

### Training XGBoost to predict Transported

To use XGBoost optimially I split the dataset into train and validation sets, on a 95/5 split, then also used stratify to make sure the distribution between class 0 and 1 was the same across the train and validation sets. And then for reproducibility we set a random state of 0. Then lastly I import the XGBoost package, to utilize the XGBoost model. Again the parameters here are from best advice from the data science community (ADD REFERENCE/EVIDENCE HERE). Plus these parameters can be optimized later.

![Screenshot: Using XGBoost](screenshots/13.png)

We can see that the model obtained an accuracy score of 83% on the training, and 80% on the validation set, meaning the model probably overfitted slightly. We can plot learning curves to visualise this, and to also see whether the model converged (stabilized) during training.

![Screenshot: Learning Curves Graph](screenshots/learning-curves.png)

The preceding graph shows that the model finds the optimal weights for the data, as further training wouldn't decrease the loss by much. The graph also shows how, near the 600 epoch the model starts to learn the training data too much. This is because the loss continues to decrease on the training data but doesn't as much for the validation data. This could be reduced with early stopping (stopping training at epoch 500), further adding to the regularization (L1 or L2), or better feature selection to prevent the curse of dimensionality (ADD A REFERENCE HERE, TO SHOW WHAT THIS IS).

### Using the XGBoost to predict Transported

So Kaggle provided two files, a train set in which we have trained the model on, and a test set which hasn't been seen yet. This test set is another csv but this time without the transported (target variable) within it. It is now the models job to predict for each of the test records (passengers) whether they were transported or not.

To make sure I obtain accurate results I first carry out all the transformations I did on the train set, on the test set. This included handling any missing values, standardizing features, encoding categorical features and so on.

After this, the last stage is to call predict on the trained model and passing in the data you want to make predictions for. See below for the predictions variable which holds an array of the predicitons.

![Screenshot: XGBoost Predictions](screenshots/15.png)

### Results and Analysis

Due to the test set coming only with the features, I cannot see how well the model did at predicting the new unseen data. However, due to using a validation set during training I got a good idea how well this model could generalize. Due to the small difference between the train and validation accuracy scores and a small gap between their learning curves, it would be fair to say that we could expect a 80%-85% accuracy score for the test set we have just predicited.

This may seem low, but when taking into account how mixed together these data points were alongs obscure patterns, I think the model has done well. Also to note the benchmark model for this task achieves 79% accruacy and this model achieve 80%+. (ADD IN REFERNCES/EVIDENCE HERE).

Therefore these results show that this model is a very strong candidate when implementing ML solutions into an business, or even updating existing ML solutions with this model if it achieves higher predictive performance than the currently implemented model. 

### Practical Applications

The techniques and models shown in this project can be applied to various business contexts to enhance decision-making processes. Here are a few examples:

- Customer Segmentation: By predicting customer behavior, businesses can tailor their marketing strategies to target specific segments more effectively. For example predicting whether custoemrs will chrun.
- Risk Management: Predictive models can help identify potential risks and mitigate them before they impact the business. For example, they could predict risk categories for different assets in investing.
- Resource Allocation: By understanding and looking at the patterns and trends the model learns, businesses can optimize resource distribution to areas with the highest potential impact.

These applications demonstrate how the insights gained from the solution to this problem can be leveraged to drive better business outcomes through data-driven decision-making.

## Recommendations for Future Iterations

1. **Incorporating Hyperparameter Tuning:** Future iterations of this project could expand from using academic suggestions for the parameters to actually using techniques like grid search or packages like Optuna which achieves the same thing. Hopefully, this would improve the accuracy of the model.

2. **Stopping overfitting:** In future iterations I would also like to reduce the amount of overfitting, by adding on the regualrization front, to increase the models generalizability.

3. **Incorporate the model into MLFlow and productionise:** Consider productionising this model with tools like MLFlow and increase work effiency by using its UI for experiments when it comes to testing other hyperparameters/models.

## Challenges Encountered

One challenge encountered during the project was trying to use and plot the validation learning curves. This was solved when I realised `model.fit` could take `eval_set` as a parameter.

Also the handling of missing values was tricky due to the mix of categorical and numerical data. What helped was realising that removing one or two rows which are proving to be difficult during data cleansing, can actually help the model generalize better.

## References

!TODO
