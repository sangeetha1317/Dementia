import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
## Read the CSV File Using Pandas read_csv function
df = pd.read_csv('./DATASET/oasis_longitudinal.csv')
# print the concise summery of the dataset
df.info()
print("Tota Rows and Columns (Rows,Columns) : ",df.shape)
#print first five rows of the dataset
df.head(10)
#print concise summery of the dataset
df.describe()
#since the dataset contain null values also
#count total rows in each column which contain null values
df.isna().sum()
#for counting the duplicate elements we sum all the rows
sum(df.duplicated())
#fill null value with their column mean and median
df["SES"].fillna(df["SES"].median(), inplace=True)
df["MMSE"].fillna(df["MMSE"].mean(), inplace=True)
#see how many people have dementia
#same person visits two or more time so only take the single visit data
sns.set_style("whitegrid")
ex_df = df.loc[df['Visit'] == 1]
sns.countplot(x='Group', data=ex_df)
#We have three groups so convert Converted Group Into Demented
ex_df['Group'] = ex_df['Group'].replace(['Converted'], ['Demented'])
41
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
sns.countplot(x='Group', data=ex_df)
# bar drawing function
def bar_chart(feature):
    Demented = ex_df[ex_df['Group']=='Demented'][feature].value_counts()
    Nondemented = ex_df[ex_df['Group']=='Nondemented'][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))
    print(df_bar)
# Gender and Group ( Female=0, Male=1)
bar_chart('M/F')
plt.xlabel('Group',fontsize=13)
plt.xticks(rotation=0,fontsize=12)
plt.ylabel('Number of patients',fontsize=13)
plt.legend()
plt.title('Gender and Demented rate',fontsize=14)
plt.figure(figsize=(10,5))
sns.violinplot(x='M/F', y='CDR', data=df)
plt.title('Violin plots of CDR by Gender',fontsize=14)
plt.xlabel('Gender',fontsize=13)
plt.ylabel('CDR',fontsize=13)
plt.show()
plt.figure(figsize=(10,5))
sns.violinplot(x='CDR', y='Age', data=df)
plt.title('Violin plot of Age by CDR',fontsize=14)
plt.xlabel('CDR',fontsize=13)
plt.ylabel('Age',fontsize=13)
plt.show()
#find the outliers in each of the column
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    42
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
list_atributes = ['MR Delay','EDUC', "SES", "MMSE", 'eTIV', "nWBV", "ASF"]
print("Outliers: \n")
for item in list_atributes:
    print(item,': ',outliers_iqr(df[item]))
from pylab import rcParams
rcParams['figure.figsize'] = 8,5
cols = ['Age','MR Delay', 'EDUC', 'SES', 'MMSE', 'CDR','eTIV','nWBV','ASF']
x=df.fillna('')
sns_plot = sns.pairplot(x[cols])
#boxplots which shows the IQR(Interquartile Range )
fig, axes = plt.subplots(2,3,figsize = (16,6))
fig.suptitle("Box Plot",fontsize=14)
sns.set_style("whitegrid")
sns.boxplot(data=df['SES'],orient="v",width=0.4,palette="colorblind",ax=axes[0]
[0]);
sns.boxplot(data=df['EDUC'],orient="v",width=0.4,palette="colorblind",ax=axes[
1]);
sns.boxplot(data=df['MMSE'],orient="v",width=0.4,palette="colorblind",ax=axes
[0][2]);sns.boxplot(data=df['CDR'], orient="v",width=0.4, palette="colorblind",ax
= axes[1][0]);sns.boxplot(data=df['eTIV'],
orient="v",width=0.4,palette="colorblind",ax =
axes[1][1]);sns.boxplot(data=df['ASF'],orient="v",width=0.4,palette="colorblind",
ax = axes[1][2]);
#xlabel("Time");
#convet the charecter data into numeric
group_map = {"Demented": 1, "Nondemented": 0}
df['Group'] = df['Group'].map(group_map)
df['M/F'] = df['M/F'].replace(['F','M'], [0,1])
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 240 , 10 , as_cmap = True )
    43
    _ = sns.heatmap(corr,cmap = cmap,square=True, cbar_kws={ 'shrink' : .9 },
    ax=ax, annot = True, annot_kws = { 'fontsize' : 12 })
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)
# Encode columns into numeric
from sklearn.preprocessing import LabelEncoder
for column in df.columns:
    le = LabelEncoder()
df[column] = le.fit_transform(df[column])
from sklearn.model_selection import train_test_split
feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV",
"ASF"]      
predicted_class_names = ['Group']
X = df[feature_col_names].values
y = df[predicted_class_names].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=42)
from sklearn import metrics
def plot_confusion_metrix(y_test,model_test):
    cm = metrics.confusion_matrix(y_test, model_test)
plt.figure(1)
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Nondemented','Demented']
roc_curves(vote_hard)
accuracy(vote_hard)
#pred = vote_hard.predict(X_test)
#accu = metrics.accuracy_score(y_test,pred)
#print("\nAcuuracy Of the Model: ",accu,"\n\n")
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X_train,
y_train.ravel())
44
vote_soft.fit(X_train, y_train.ravel())
report_performance(vote_soft)
roc_curves(vote_soft)
accuracy(vote_soft)
#pred = vote_soft.predict(X_test)
#accu = metrics.accuracy_score(y_test,pred)
#print("\nAcuuracy Of the Model: ",accu,"\n\n")
clfs =[ExtraTreesClassifier(),GradientBoostingClassifier(),AdaBoostClassifier()]
for model in clfs:
    print(str(model).split('(')[0],": ")
model.fit(X_train,y_train.ravel())
X = pd.DataFrame(X_train)
report_performance(model)
roc_curves(model)
accuracy(model)
for i in total_fpr.keys():
    plt.plot(total_fpr[i],total_tpr[i],lw=1, label=i)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend()
