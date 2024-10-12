# %%
import pandas as pd
import numpy as np
import seaborn as sns 
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
import shap
import lime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# %% [markdown]
# ## Weather Prediction Project
# This project aims to predict whether it will rain tomorrow based on historical weather data. Using a dataset with various meteorological features such as temperature, humidity, wind conditions, and cloud cover, a machine learning model is developed to forecast rainfall. The project involves data preprocessing, feature selection, and model training to achieve accurate predictions. Models such as logistic regression, decision trees, and random forests are explored, with performance evaluated through metrics like accuracy, precision, and recall. The goal is to provide a reliable, data-driven tool for weather forecasting.

# %% [markdown]
# ## 1. Data input

# %% [markdown]
# ### 1.1 Reading data from a file

# %%
data = pd.DataFrame()
data = pd.read_csv("weather.csv")

data.head(10)

# %% [markdown]
# ### 1.1 Data description

# %%
data.describe()

# %% [markdown]
# ## 2. Data cleaning
# ### 2.1 Type checking

# %%
data.dtypes

# %% [markdown]
# ### 2.2 Conversion to numerical values
# We can see, that some of the columns are of type **object** **(String)**, however we want all our columns to be numercial.

# %%
object_columns = data.select_dtypes(include='object').columns
data[object_columns].head()

# %% [markdown]
# First three columns contain types of *wind directions* ***(altogether 16 possible types)*** and columns containing information about ***rain***
# are of type *boolean* (yes or no).
# 
# We can create conversion mappings for those columns.

# %%
wind_to_float_mapping = {"N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE" : 67.5, 
                         "E":90.0, "ESE" : 115.5, "SE":135.0, "SSE" : 157.5, 
                         "S":180.0, "SSW": 202.5, "SW": 225.0, "WSW":247.5, 
                         "W":270, "WNW": 292.5, "NW": 315.0, "NNW": 337.5 }

bool_to_float_mapping = {'Yes':1, 'No':0}


# %% [markdown]
# ### 2.3 Wind columns
# 
# We can filter out *wind related* columns and apply mapping to degrees, where **N = 0°, S = 180°, SE = 135°,  NNE = 22.5°, etc.**

# %%
wind_dir_columns = data[object_columns].filter(like='Wind').columns
data[wind_dir_columns] = data[wind_dir_columns].replace(wind_to_float_mapping)
data[wind_dir_columns].head()

# %%
data[wind_dir_columns].dtypes

# %% [markdown]
# ### 2.4 Yes or No columns
# 
# We can select **Rain related** columns the same way and also apply mapping, after successfully converting **wind related** ones

# %%
yes_or_no_columns = data.filter(like="Rain").columns
data[yes_or_no_columns] = data[yes_or_no_columns].replace(bool_to_float_mapping)

# %% [markdown]
# ### 2.5 Checking if all datatypes are numerical

# %%
data.dtypes

# %% [markdown]
# ### 2.6 Checking for duplicates

# %%
data.duplicated().sum()

# %% [markdown]
# ### 2.7 Checking for N/A values

# %%
data.isnull().sum()

# %% [markdown]
# ### 2.8 Replacing N/As with the **median** of a given column

# %%
for col in data.columns:
    data[col] = data[col].fillna(data[col].median())

data.isnull().sum()

# %% [markdown]
# ### 2.8.1 Dropping rows with any N/As present is also a possible solution

# %%
# data = data.dropna()

# %% [markdown]
# ## 3. Building a correlation matrix
# 
# This ***22x22*** matrix allows us to determine columns, that have the ***highest correlation*** (have the most impact) on a result column **("RainTommorow)**

# %%
corr_matrix = data.corr()

# %%
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix,annot=True)

# %% [markdown]
# ### 3.1 Correlation for RainTomorrow collumn 

# %%
corr_list = corr_matrix["RainTomorrow"].abs().sort_values(ascending=False)
corr_list

# %% [markdown]
# We should delete the ***RainTomorrow*** column from the list, as it's correlation to itself is trivially equal to ***1.0***
# 
# We can then select ***five most impotant columns*** for the next stages

# %%
corr_list.pop("RainTomorrow")
important_columns = corr_list[0:5].index
important_columns

# %% [markdown]
# ## 4. Data visualization

# %% [markdown]
# ### 4.1 Rain depending on various parameteres, such as temperature, evaporation, sunshine, etc.
# Blue dots - ***no rain***, orange dots - ***rain***

# %%
plt.figure(figsize=(20,10))
sns.jointplot(data=data, x="MaxTemp", y="MinTemp", hue="RainToday",height=8,s=30,alpha=0.7)
sns.jointplot(data=data, x="MinTemp", y="MaxTemp", hue="RainToday",height=8,s=30,alpha=0.7, )
sns.jointplot(data=data, x="MinTemp", y="Rainfall", hue="RainToday",height=8,s=30,alpha=0.7)
sns.jointplot(data=data, x="MaxTemp", y="Evaporation", hue="RainToday",height=8,s=30,alpha=0.7)
sns.jointplot(data=data, x="Rainfall", y="Evaporation", hue="RainToday",height=8,s=30,alpha=0.7)
sns.jointplot(data=data, x="Sunshine", y="Cloud9am", hue="RainToday",height=8,s=30,alpha=0.7)
sns.jointplot(data=data, x="Cloud9am", y="Cloud3pm", hue="RainToday",height=8,s=30,alpha=0.7)

# %% [markdown]
# ## 5. Preparing learning models

# %% [markdown]
# ### 5.1 'RISK_MM' Column

# %%
print(important_columns)

# %% [markdown]
# From the list of ***important columns***, we can see, that **'RISK_MM'** is heavily corelated to **'RainTomorrow'**, which is not really a surprise, while **'RISK_MM'** describes the predicted amount of rainfall tomorrow. In case it's more than 0 - rain tomorrow is obviously more likely. 
# 
# **DecisionTreeClassifier()** clearly indicates, that this column alone heavily influences estimator's behaviour.

# %%
X = data.drop(['RainTomorrow'], axis=1)  
y = data['RainTomorrow']               

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

tree1 = DecisionTreeClassifier()
tree1.fit(X_train, y_train)
y_pred = tree1.predict(X_test)
accuracy_score(y_pred, y_test)

# %% [markdown]
# ### This model's accuracy is equal to **1**, which is a little suspicious

# %%
plot_tree(tree1, filled=True, feature_names=X_train.columns)
plt.show()

# %% [markdown]
# We can see, that algorithm ***only*** takes into consideration this parameter, however the goal of our experiment is to not use other predictions, but rather create our own.
# 
# #### So we should delete this column from the dataset

# %%
corr_list.pop("RISK_MM")
important_columns = corr_list[0:5].index
important_columns

# %% [markdown]
# ### 5.2 Building the model

# %% [markdown]
# As a part of this experiment, we are going to train our models on the whole dataset, as well as columns, we determined as 'important' (columns with the highest correlation to **'RainTomorrow'**) 

# %%
### Training data on all columns
X = data.drop(['RainTomorrow', 'RISK_MM'], axis=1)  
### Training data on 5 columns with the highest correlation to y
X_important = data[important_columns]     
### Result column
y = data['RainTomorrow']                  

# %% [markdown]
# ### 5.3 Splitting the dataset: 80% - training, 20% - testing

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# %%
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split (
    X_important, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### 5.4 Scaling both datasets

# %%
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler1 = StandardScaler()
scaler1.fit(X_train_imp)

X_train_scaled_imp = scaler1.transform(X_train_imp)
X_test_scaled_imp = scaler1.transform(X_test_imp)


# %% [markdown]
# ## 6. Classifiers

# %% [markdown]
# ### 6.1 Logistic Regression

# %%
lr = LogisticRegression(max_iter=1000, solver='lbfgs')
lr.fit(X_train_scaled,y_train)

# %%
y_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred)
lr_acc

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### 6.2 K-Nearest Neighbors

# %% [markdown]
# Iterating through different amount of neighbors allows us to find the optimal ***n_neighbors*** parameter, at the same time avoiding overfitting

# %%
knn_models = {}

for i in range(3, 101):
    knn_models[f'knn_{i}'] = KNeighborsClassifier(n_neighbors=i).fit(X_train_scaled, y_train)

# %%
scores = [()]
for knn in knn_models:
    y_pred = knn_models[knn].predict(X_test_scaled)
    scores.append((accuracy_score(y_test, y_pred), knn))

print(max(scores)[1])
knn_best = knn_models[max(scores)[1]]


# %%
y_pred_best = knn_best.predict(X_test_scaled)

knn_acc = accuracy_score(y_test, y_pred_best)
knn_acc

# %%
print(classification_report(y_test, y_pred_best))

# %% [markdown]
# ### 6.3 Decision Tree Classifier

# %%
tree = DecisionTreeClassifier(
    max_depth=None,            # No limit on tree depth (let it grow fully)
    min_samples_split=5,       # Default value, but you can increase it if needed
    min_samples_leaf=5,        # Default value, reduce to consider more splits
    max_features=None          # Use all features when splitting a node
)
tree.fit(X_train, y_train)

# %%
y_pred = tree.predict(X_test)
tree_acc = accuracy_score(y_test, y_pred)
tree_acc

# %%
from sklearn.tree import export_graphviz
import graphviz

classes = ['No rain', "Rain"]
dot_data = export_graphviz(
    tree, 
    out_file=None, 
    feature_names=X_train.columns,
    class_names=classes, 
    filled=True, 
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree") 

graph

# %% [markdown]
# ### 6.4 Random Forest Classifier

# %%
forest = RandomForestClassifier( 
    max_depth=None,            
    min_samples_split=5,       
    min_samples_leaf=5,        
    max_features=None          
)
forest.fit(X_train, y_train)

# %%
y_pred = forest.predict(X_test)
forest_acc = accuracy_score(y_pred, y_test)
forest_acc

# %% [markdown]
# ### 6.5 We can create so called *'ensemble'* of classifiers, choosing 'soft' voting method (majority of classifiers decides about the output)

# %%
voting_classifier = VotingClassifier(
    estimators=[('tree', tree),  ('knn', knn_best), ('lr', lr), ('rf', forest)],  
    voting='soft'
)

# %%
voting_classifier.fit(X_train_scaled, y_train)
y_pred = voting_classifier.predict(X_test_scaled)

voting_acc = accuracy_score(y_pred, y_test)
voting_acc

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## 7. Neural network

# %%
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  

early_stopping = EarlyStopping(
    monitor='accuracy',        # Monitor validation loss
    patience=10,               # Stop training after 10 epochs with no improvement
    restore_best_weights=True  # Restore the best weights after stopping
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.1, callbacks=early_stopping,) 
network_loss, network_acc = model.evaluate(X_test_scaled, y_test)
model

print(f"Test Accuracy: {network_acc}")

# %% [markdown]
# ## 8. Results

# %% [markdown]
# ### 8.1 Acuracies of all previously tested classifiers
# 

# %%
accuracies = [lr_acc, knn_acc, tree_acc, forest_acc, voting_acc, network_acc]
accuracies
    

# %% [markdown]
# ### 8.2 All atributes vs 'Important' atributes
# 
# We're iterating through all previously used classifiers as well as two datasets, creating a Data Frame for comparing different statistical parameters 

# %%
classifiers = {
    'Random Forest': forest,
    'Logistic Regression': lr,
    'K-Nearest Neighbors   ': knn_best,
    'Decision Tree': tree,
    'Voting Classifier' : voting_classifier
}

datasets = {
    'All atributes': (X_train_scaled, X_test_scaled, y_train, y_test),
    '\'Important atributes\'': (X_train_scaled_imp, X_test_scaled_imp, y_train_imp, y_test_imp)
}

feature_columns = {'All atributes' : X.columns,  '\'Important atributes\'': X_train_imp.columns}

def create_neural_network(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

results = []

for dataset_name, (X_train_1, X_test_1, y_train_1, y_test_1) in datasets.items():
    input_shape = X_train_1.shape[1]  
    for clf_name, clf in classifiers.items():
        
        clf.fit(X_train_1, y_train_1)
        y_pred_1 = clf.predict(X_test_1)
        
        accuracy = accuracy_score(y_test_1, y_pred_1)
        precision = precision_score(y_test_1, y_pred_1)
        recall = recall_score(y_test_1, y_pred_1)
        f1 = f1_score(y_test_1, y_pred_1)
        
        results.append({
            'Dataset': dataset_name,
            'Classifier': clf_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    

    nn_model = create_neural_network(input_shape)

    early_stopping = EarlyStopping(
        monitor='accuracy',     
        patience=10,             
        restore_best_weights=True  
        )
    
    nn_model.fit(X_train_1, y_train_1, epochs=50, batch_size=16, verbose=0, validation_split=0.1, callbacks=early_stopping)
    
    y_pred_nn = (nn_model.predict(X_test_1) > 0.5).astype("int32") 
    accuracy_nn = accuracy_score(y_test_1, y_pred_nn)
    precision_nn = precision_score(y_test_1, y_pred_nn)
    recall_nn = recall_score(y_test_1, y_pred_nn)
    f1_nn = f1_score(y_test_1, y_pred_nn)
    
    results.append({
        'Dataset': dataset_name,
        'Classifier': 'Neural Network',
        'Accuracy': accuracy_nn,
        'Precision': precision_nn,
        'Recall': recall_nn,
        'F1-Score': f1_nn
    })

# %% [markdown]
# ### 8.3 Created Data Frame

# %%
results_df = pd.DataFrame(results)
results_df.sort_values(['Classifier'], inplace=True)
results_df

# %% [markdown]
# ### 8.4 Data visualization

# %%
results_df.pivot(index='Classifier', columns='Dataset', values='Accuracy').plot(kind='bar', figsize=(10, 6))
plt.title('Accuracy Comparison Across Classifiers and Datasets')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0.6, 0.9)
plt.show()


# %%
results_df.pivot(index='Classifier', columns='Dataset', values='Precision').plot(kind='bar', figsize=(10, 6))
plt.title('Precission Comparison Across Classifiers and Datasets')
plt.ylabel('Precission')
plt.xticks(rotation=45)
plt.ylim(0.39, 0.9)
plt.show()


# %% [markdown]
# ## 9. Explainable AI Elements

# %% [markdown]
# 
# **Explainable AI (XAI)** helps users understand and trust machine learning models by making their decisions transparent. XAI is essential for improving model transparency, accountability, trust, and fairness, especially when working with complex "black box" models like deep neural networks.
# 
# ---
# 
# ### LIME (Local Interpretable Model-Agnostic Explanations)
# 
# **LIME** explains individual predictions of any model by approximating the complex model locally with a simpler, interpretable model (like linear regression). It perturbs the input data, observes how the model's prediction changes, and builds a linear model to explain the influence of each feature on that specific prediction.
# 
# - **Advantages**: Simple, model-agnostic, provides local explanations for individual predictions.
# - **Limitations**: Provides only local approximations, and results can vary depending on the data perturbations.
# 
# ---
# 
# ### SHAP (SHapley Additive exPlanations)
# 
# **SHAP** is based on Shapley values from game theory and fairly attributes the contribution of each feature to a model's prediction. It provides both global and local explanations by showing how much each feature impacts the prediction, using a solid theoretical foundation.
# 
# - **Advantages**: Provides consistent, globally interpretable feature importance. Handles feature interactions well.
# - **Limitations**: Computationally expensive, especially for large models and datasets.
# 
# ---
# 
# ### LIME vs. SHAP
# 
# - **LIME**: Quick, local, and easy-to-interpret but only provides approximations.
# - **SHAP**: More theoretically sound, offers both global and local explanations, but can be computationally heavy.
# 
# 

# %% [markdown]
# ### 9.1 LIME Explanation

# %%
datasets1 = {
    'All atributes': (X_train, X_test, y_train, y_test),
    '\'Important atributes\'': (X_train_imp, X_test_imp, y_train_imp, y_test_imp)
}

# %%
def plot_lime_exp(fig, fig_index, exp_data, title):
    exp_data = dict(sorted(exp_data.items(), key=lambda item: abs(item[1]), reverse=True))
    
    features = list(exp_data.keys())[::-1]  
    explanations = list(exp_data.values())[::-1]  

    ax = fig.add_subplot(fig_index)  
    lime_bar = ax.barh(features, explanations)  
    ax.set_title(title, fontsize=15)

    color_list = sns.color_palette("husl", len(exp_data))  
    for i, bar in enumerate(lime_bar):
        bar.set_color(color_list[i % len(color_list)])  

    ax.set_facecolor('white')
    plt.box(False)


for dataset_name, (X_train, X_test, y_train, y_test) in datasets1.items():
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_shape = X_train_scaled.shape[1]
    keras_model = create_neural_network(input_shape)

    keras_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

    def keras_predict_proba(X_unscaled):
        X_scaled = scaler.transform(X_unscaled) 
        pred = keras_model.predict(X_scaled)  
        return np.hstack([1 - pred, pred]) 

    classifiers['Neural Network (Keras)'] = keras_predict_proba

    fig = plt.figure(figsize=(15, 15))
    plt.suptitle(f"LIME Explanations for {dataset_name}", fontsize=20)  
    
    instance = X_test.iloc[0].values.reshape(1, -1)  
  
    for idx, (name, clf) in enumerate(classifiers.items(), start=1):
        if name == 'Neural Network (Keras)':  
            predict_fn = keras_predict_proba 
        else:
            clf.fit(X_train_scaled, y_train)  
            predict_fn = lambda X: clf.predict_proba(scaler.transform(X))  

        X_train_np = X_train.values  
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_np,  
            feature_names=feature_columns[dataset_name],
            class_names=['No Rain', 'Rain'],
            mode='classification'
        )
    
        exp = explainer.explain_instance(
            data_row=instance[0],  
            predict_fn=predict_fn
        )
        
       
        exp_data = dict(exp.as_list())
        plt.subplots_adjust(wspace=0.8, hspace=0.35)

        plot_lime_exp(fig, 320 + idx, exp_data, f"{name}")



plt.show()


# %% [markdown]
# ## Value interpretation for LIME
# 
# Numerical value represents each feature's importance for overall classification for each model. If the value is positive - it increases the likelihood of rain, if negative - decreases it.
# 
# ### For example:
# **Pressure3pm - 1017.5 < Pressure <= 1021.8** - pressure in this range decreases the probability of rain more significantly, than other parameters
# 
# **Humidity9am - 72 < Humidity <= 81** - humidity in this relatively high range increases the probability of rain more significantly, than other parameters

# %%
for dataset_name, (X_train_scaled, X_test_scaled, y_train, y_test) in datasets.items():
    knn_best.fit(X_train_scaled, y_train)
    
    background = shap.kmeans(X_train_scaled, 100) 
    explainer = shap.KernelExplainer(knn_best.predict_proba, background)
    
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure()
    plt.title(f"SHAP Summary Plot for {dataset_name}")
    shap.summary_plot(shap_values[:, :, 1], X_test_scaled,   feature_names=feature_columns[dataset_name], plot_type='violin')
    plt.show()



# %% [markdown]
# ## Value interpretation for SHAP
# On x-axis every single value is represneted. It has three main parameteres:
# - ***How far*** the value is on the axis represents how ***strongly*** it influences the outcome
# - If the value is ***positive*** - probability of **rain is increased**, if ***negative*** - **decreased**
# - ***Colour*** represents if the value is **high or low** (relative to it's ***median*** value)
# ### For example:
# 
# For **Pressure** at 3pm
# - The majority of points is on the left side of the plot, so it's rather not causing *rain*
# - The further to the left side - the higher values for the pressure are (colour is changing from blue to pink), so the higher the value - the stronger it influences the class 
# - On the positive side - the values are mainy low (blue), which means, that low pressure is corresponding to a higher likelihood of *rain* 
# - Pressure3pm feature has a continuous range of around 0.3 - the most of all features, which means, that it behaves pretty consistently
# 
# 
# For **Cloud:** at 3pm:
# - The  points are distributed relatively equally on both sides of the plot, so this feature influences the outcome both ways
# - Values are gradually changing from left to right and low to high (blue to pink), there are no outliers
# - Low amount of clouds is realted to the lack of rain (the lower the stronger), and high amount corresponds to high chance of rain (the higher the stronger)
# 


