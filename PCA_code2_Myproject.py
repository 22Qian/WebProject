# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:08:57 2025

@author: Martian
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:26:40 2025

@author: Martian
"""

####-----------------------------------------------------------
#### PCA and Pairwise Correlation
##   3D Scatterplot
##   !! Dataset !!  (A sample of this data was used in this code)
##   Link to Full Iris Dataset from Kaggle:
##   https://www.kaggle.com/datasets/uciml/iris?resource=download
##
## Gates, 2024
####------------------------------------------------------------
##
## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import seaborn as sns


path="D:/OM_PhD_CUBoulder/Classes/25S_MachineLearning/M2/PCA/Hotel1first150r.csv"
DF=pd.read_csv(path)
print(DF)
OriginalDF=DF.copy()
print(OriginalDF)
columns = list(DF.columns)  # Store column names in a list
print(columns)
Pricecolumns_to_scale = ["Nightly_Rate"]
DF[Pricecolumns_to_scale] = DF[Pricecolumns_to_scale] * 0.001
##--------------------------------
## Remove and save the label
## Next, update the label so that 
## rather than names like "Iris-setosa"
## we use numbers instead. 
## This will be necessary when we "color"
## the data in our plot
##---------------------------------------
DFLabel1=DF["Distribution_Channel"]  ## Save the Label 
print(DFLabel1)  ## print the labels
print(type(DFLabel1))  ## check the datatype you have
DFLabel_string=DFLabel1
## Remap the label names from strongs to numbers
MyDic={"CRO/Hotel":0, "GDS":1, "WEB":2}
DFLabel1 = DFLabel1.map(MyDic)  ## Update the label to your number remap values
print(DFLabel1) ## Print the labels to confirm 

DFLabel2=DF["Purchased_Rate_Code"]  ## Save the Label 
print(DFLabel2)  ## print the labels
print(type(DFLabel2))  ## check the datatype you have
DFLabel_string=DFLabel2
## Remap the label names from strongs to numbers
MyDic={"Rate 1":1, "Rate 2":2, "Rate 8":8}
DFLabel2 = DFLabel2.map(MyDic)  ## Update the label to your number remap values
print(DFLabel2) ## Print the labels to confirm how

## Now, remove the label from the original dataframe
DF = DF.drop(columns=["Distribution_Channel", "Purchased_Rate_Code"])

print(DF) #Print the dataframe to confirm 

###-------------------------------------------
### Standardize your dataset
###-------------------------------------------
# Assuming DF is a DataFrame
scaler = StandardScaler()  # Instantiate scaler
scaled_array = scaler.fit_transform(DF)  # Scale the data

# Convert back to DataFrame and retain column names
DF_scaled = pd.DataFrame(scaled_array, columns=DF.columns)

print(DF_scaled)
print(type(DF_scaled))  # Now it's a pandas DataFrame
print(DF_scaled.shape)

# Plot histograms for all features
plt.figure(figsize=(20, 20))
for i, column in enumerate(DF_scaled.columns):  # Iterate over DataFrame columns
    plt.subplot(5, 5, i + 1)
    plt.hist(DF_scaled[column], bins=20, color='grey', alpha=0.7)
    plt.title(column)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
plt.tight_layout()
plt.show()
plt.savefig("histograms_features.png", dpi=300, bbox_inches='tight')  # Save as a high-resolution PNG



DF_scaled.to_csv("Hotel1first150r_scaled.csv")
print(columns)
###############################################
###--------------PERFORM PCA------------------
###############################################

##PCA with n_components = 2
## Instantiate PCA and choose how many components
MyPCA2=PCA(n_components=2)
# Project the original data into the PCA space
Result=MyPCA2.fit_transform(DF)
## Print the values of the first component 
#print(Result[:,0]) 
print(Result) ## Print the new (transformed) dataset
print("The eigenvalues:", MyPCA2.explained_variance_)
## Proof
MyCov=np.cov(Result.T)
print("Covar of the PC PCA Matrix: \n", MyCov) ## The variance here (on the diagonal) will match the eigenvalues
print("The relative eigenvalues are:",MyPCA2.explained_variance_ratio_)
print("The actual eigenvalues are:", MyPCA2.explained_variance_)
EVects=MyPCA2.components_
print("The eigenvectors are:\n",EVects)
print(type(EVects))
print("The shape of the eigenvector matrix is\n", EVects.shape)
print(DF.shape)
print(DF)
## Proof to transform origial data to eigenbasis
## using the eigenvectors matrix.
print(EVects.T)
##  (15, 4) @ (4, 3)
Transf=DF@EVects.T
print("Proof that the transformed data is the EVects @ Data\n",Transf)
# Create a scatter plot of PCA-transformed data
plt.figure(figsize=(10, 6))
plt.scatter(Result[:, 0], Result[:, 1], alpha=0.7, c='blue', edgecolors='k')

# Add labels and title
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA Projection: Data Transformed into 2D Space")

# Draw axis lines for reference
plt.axhline(y=0, color='gray', linestyle='--', linewidth=2.2)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=2.2)

# Show the plot
plt.grid(True)
plt.show()

# Get explained variance ratio
explained_variance_ratio = MyPCA2.explained_variance_ratio_

# Print explained variance for each PC
print("Explained Variance Ratio for Each Principal Component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.2%}")

# Compute total variance retained
total_variance_retained = np.sum(explained_variance_ratio)
print(f"\nTotal Variance Retained (by first 2 PCs): {total_variance_retained:.2%}")

#Eigenvalues
# Compute cumulative eigenvalues
ACCUM_eigenvalues = np.cumsum(MyPCA2.explained_variance_)
print("Cumulative Eigenvalues:", ACCUM_eigenvalues)
print("Explained Variance Ratio:", MyPCA2.explained_variance_ratio_)

# Create figure
fig3 = plt.figure(figsize=(8,6))

# Plot bar chart for explained variance ratio
bars = plt.bar(range(0, len(MyPCA2.explained_variance_ratio_)), 
               MyPCA2.explained_variance_ratio_, 
               alpha=0.5, align='center', label='Individual Explained Variances')

# Annotate each bar with the variance ratio value
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2%}', 
             ha='center', va='bottom', fontsize=10, color='black')

# Labels and title
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title("Eigenvalues: Percentage of Variance/Information")

# Save the figure **before** plt.show()
plt.savefig("Eigenvalues2D.png", dpi=300, bbox_inches='tight')  # High-resolution PNG

plt.show()


##n=3
## Instantiate PCA and choose how many components
MyPCA3=PCA(n_components=3)
# Project the original data into the PCA space
Result=MyPCA3.fit_transform(DF)
## Print the values of the first component 
#print(Result[:,0]) 
print(Result) ## Print the new (transformed) dataset
print("The eigenvalues:", MyPCA3.explained_variance_)
## Proof
MyCov=np.cov(Result.T)
print("Covar of the PC PCA Matrix: \n", MyCov) ## The variance here (on the diagonal) will match the eigenvalues
print("The relative eigenvalues are:",MyPCA3.explained_variance_ratio_)
print("The actual eigenvalues are:", MyPCA3.explained_variance_)
EVects=MyPCA3.components_
print("The eigenvectors are:\n",EVects)
print(type(EVects))
print("The shape of the eigenvector matrix is\n", EVects.shape)
print(DF.shape)
print(DF)
## Proof to transform origial data to eigenbasis
## using the eigenvectors matrix.
print(EVects.T)
##  (15, 4) @ (4, 3)
Transf=DF@EVects.T
print("Proof that the transformed data is the EVects @ Data\n",Transf)

# Perform PCA with 3 components

# Create a 3D Scatter Plot
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

# Plot PCA-transformed data
ax.scatter(Result[:, 0], Result[:, 1], Result[:, 2], alpha=0.7, c='blue', edgecolors='k')

# Label axes
ax.set_xlabel("Principal Component 1 (PC1)", fontsize=12, labelpad=15)
ax.set_ylabel("Principal Component 2 (PC2)", fontsize=12, labelpad=15)
ax.set_zlabel("Principal Component 3 (PC3)", fontsize=12, labelpad=15)
ax.set_title("PCA Projection in 3D Space", fontsize=14)

# Adjust layout and view
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.80)  # Adjust margins
ax.view_init(elev=20, azim=25)  # Adjust viewing angle

# Save the figure before showing it
plt.savefig("PCA_3D_plot.png", dpi=300, bbox_inches='tight')

# Get explained variance ratio
explained_variance_ratio = MyPCA3.explained_variance_ratio_

# Print explained variance for each PC
print("Explained Variance Ratio for Each Principal Component:")
for i, var in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {var:.2%}")

# Compute total variance retained
total_variance_retained = np.sum(explained_variance_ratio)
print(f"\nTotal Variance Retained (by first 3 PCs): {total_variance_retained:.2%}")


##Eigenvalues
# Compute cumulative eigenvalues
ACCUM_eigenvalues = np.cumsum(MyPCA3.explained_variance_)
print("Cumulative Eigenvalues:", ACCUM_eigenvalues)
print("Explained Variance Ratio:", MyPCA3.explained_variance_ratio_)

# Create figure
fig4 = plt.figure(figsize=(8, 6))

# Plot bar chart for explained variance ratio
bars = plt.bar(range(0, len(MyPCA3.explained_variance_ratio_)), 
               MyPCA3.explained_variance_ratio_, 
               alpha=0.5, align='center', label='Individual Explained Variances')

# Annotate each bar with the variance ratio value
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2%}', 
             ha='center', va='bottom', fontsize=10, color='black')

# Labels and title
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title("Eigenvalues: Percentage of Variance/Information")

# Save the figure **before** plt.show()
plt.savefig("Eigenvalues3D.png", dpi=300, bbox_inches='tight')  # High-resolution PNG

plt.show()


################################################
# Extract loadings
## Rerun PCA with all columns - no dim reduction
############################################################
MyPCAall=PCA(n_components=7)
# Project the original data into the PCA space
Result2=MyPCAall.fit_transform(DF)
print(MyPCAall.components_) 
print(MyPCAall.components_.T) ## Makes the eigenvectors columnwise
print(MyPCAall.explained_variance_ratio_) 

print(OriginalDF)
print(len(OriginalDF))
print(OriginalDF.columns[0:4])

for i in range(1, len(OriginalDF[0:4].columns)):
               print(i)
               
#loadings = pd.DataFrame(MyPCAall.components_.T, 
                        #columns=[f'PC{i}' for i in range(1, len(OriginalDF[0:4].columns))], 
                        #index=OriginalDF.columns[0:4])

# Fix the loadings DataFrame
loadings = pd.DataFrame(MyPCAall.components_.T, 
                        columns=[f'PC{i}' for i in range(1, MyPCAall.n_components_ + 1)], 
                        index=OriginalDF.columns[:DF.shape[1]])  # Ensure matching dimensions
print(loadings)

## Print the most important variables using a threshold
threshold = 0.8
# Find features with loadings above the threshold for each principal component
important_features = {}
for column in loadings.columns:
    important_features[column] = loadings.index[loadings[column].abs() > threshold].tolist()

# Now 'important_features' dictionary contains the important features for each PC
for pc, features in important_features.items():
    print(f"{pc}: {', '.join(features)}")

# Plot heatmap of loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm')
plt.title('Feature Importance in Principal Components')
plt.show()
# Save the figure **before** displaying it
plt.savefig("heatmap_PCA.png", dpi=300, bbox_inches='tight')  # High-resolution PNG

######################Accumulated 95% PCA data
# Perform PCA on the dataset with all components
MyPCAall = PCA(n_components=7)
Result = MyPCAall.fit_transform(DF)

# Compute explained variance and accumulated variance
explained_variance = MyPCAall.explained_variance_ratio_
accumulated_variance = np.cumsum(explained_variance)

# Print variance information
print("The eigenvalues:", MyPCAall.explained_variance_)
print("Accumulated explained variance ratio:", accumulated_variance)

# Create figure
plt.figure(figsize=(10, 6))

# Barplot of explained variance ratio
bars = plt.bar(range(1, len(explained_variance) + 1), 
               explained_variance, alpha=0.5, align='center', color='grey')

# Draw a horizontal reference line at 3% variance threshold
plt.axhline(y=0.03, color='r', linestyle='--', label='3% Threshold')

# Annotate each bar with the explained variance percentage
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', 
             ha='center', va='bottom', fontsize=10, color='black')

# Labels and title
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title('Explained Variance Ratio by Principal Component')
plt.xticks(range(1, len(explained_variance) + 1))  # Ensure x-axis ticks match components

plt.legend()
plt.tight_layout()

# Save the figure **before** plt.show()
plt.savefig("PCA_Explained_Variance.png", dpi=300, bbox_inches='tight')  

plt.show()




###############################################
## Create a biplot of the most important features
##################################################
# PCA dataset (PC-transformed data)
PCA_dataset = Result  # Result from PCA transformation

# Loadings matrix (eigenvectors as columns)
EVectors_as_columns = MyPCAall.components_.T  # Ensure correct PCA loadings



# PCA Variance Explained
explained_variance_ratio = [0.54708732, 0.69284173, 0.83527006, 0.92872453, 0.97495432, 0.99999691, 1.0]
components = range(1, len(explained_variance_ratio) + 1)

plt.figure(figsize=(8,5))
plt.plot(components, explained_variance_ratio, marker='o', linestyle='-', label="Accumulated Variance")
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% Threshold')

plt.xlabel('Number of Principal Components')
plt.ylabel('Accumulated Explained Variance')
plt.title('Scree Plot: Accumulated Explained Variance')
plt.grid(True)
plt.legend()
plt.show()

###############################################
## Create a DF of the most important features
##################################################
shape= MyPCAall.components_.shape[0]
#print(shape)
feature_names=['Advance_Purchase', 'Nightly_Rate', 'Distribution_Channel', 'Purchased_Rate_Code', 'Temperature_Max', 'Temperature_Min', 'Precipitation', 'Snowfall', 'Wind_Speed']
# Find the most important feature for each principal component
most_important = [np.abs(MyPCAall.components_[i]).argmax() for i in range(shape)]
most_important_names = [feature_names[most_important[i]] for i in range(shape)]

# Build a dictionary of the most important features by PC

shape= MyPCAall.components_.shape[0]

most_important = [np.abs(MyPCAall.components_[i]).argmax() for i in range(shape)]
most_important_names = [feature_names[most_important[i]] for i in range(shape)]

# Build a doctionary of the imprtant features by PC
MyDic = {'PC{}'.format(i): most_important_names[i] for i in range(shape)}

# build the dataframe
Important_DF = pd.DataFrame(MyDic.items())
print(Important_DF)