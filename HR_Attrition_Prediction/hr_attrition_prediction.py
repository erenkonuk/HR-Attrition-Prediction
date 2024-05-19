import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

df = pd.read_csv("HR_capstone_dataset.csv")
print(df.head())
print(df.info())
print(df.describe().T)
print(df.shape)

#Aykırı değer tespiti
def detect_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers

outlier_detection = {}
features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
for feature in features:
    outlier_detection[feature] = detect_outliers(df, feature)

for feature in features:
    print(f"{feature}: {outlier_detection[feature].shape[0]} aykırı değer")

cleaned_df = df.drop(outlier_detection['time_spend_company'].index)
print(cleaned_df.shape)


#Veri Dönüşümü
categorical_features = ['Department', 'salary']
encoder = OneHotEncoder(sparse_output=False)
categorical_encoded = encoder.fit_transform(cleaned_df[categorical_features])
categorical_encoded = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))

cleaned_df = pd.concat([cleaned_df.drop(categorical_features, axis=1), categorical_encoded], axis=1)
print(cleaned_df.head().T)
print(cleaned_df.shape)


#Veri Görselleştirme
plt.figure(figsize=(15, 4))
sns.violinplot(x=cleaned_df['left'], y=cleaned_df['satisfaction_level'])
plt.title('İşten Ayrılma Durumuna Göre Memnuniyet Seviyesi')
plt.xlabel('İşten Ayrıldı (0: Hayır, 1: Evet)')
plt.ylabel('Memnuniyet Seviyesi')
plt.show()

plt.figure(figsize=(15, 4))
sns.boxplot(x='left', y='average_montly_hours', data=cleaned_df)
plt.title('İşten Ayrılma Durumuna Göre Aylık Ortalama Çalışma Saatleri')
plt.xlabel('İşten Ayrıldı (0: Hayır, 1: Evet)')
plt.ylabel('Aylık Ortalama Saatler')
plt.show()

plt.figure(figsize=(15, 4))
sns.countplot(x='number_project', hue='left', data=cleaned_df)
plt.title('Proje Sayısına Göre İşten Ayrılma Durumları')
plt.xlabel('Proje Sayısı')
plt.ylabel('Sayı')
plt.show()

plt.figure(figsize=(15, 4))
sns.scatterplot(x='last_evaluation', y='satisfaction_level', hue='left', data=cleaned_df)
plt.title('Değerlendirme Seviyesi ve Memnuniyet Seviyesine Göre İşten Ayrılma Dağılım Grafiği')
plt.xlabel('Değerlendirme Seviyesi')
plt.ylabel('Memnuniyet Seviyesi')
plt.show()

plt.figure(figsize=(15, 4))
sns.histplot(x='satisfaction_level', hue='left', data=cleaned_df, kde=True)
plt.title('Memnuniyet Seviyesi ve İşten Ayrılma')
plt.xlabel('Memnuniyet Seviyesi')
plt.ylabel('Frekans')
plt.show()


#Model performansını artırmak için yeni özellikler
cleaned_df['high_work_hours'] = (cleaned_df['average_montly_hours'] > 220).astype(int)
cleaned_df['performance_score'] = cleaned_df['satisfaction_level'] * cleaned_df['last_evaluation']
cleaned_df['work_load'] = cleaned_df['number_project'] / cleaned_df['average_montly_hours']
cleaned_df['satisfaction_evaluation_diff'] = cleaned_df['last_evaluation'] - cleaned_df['satisfaction_level']
cleaned_df['accident_hour_ratio'] = cleaned_df['Work_accident'] / cleaned_df['average_montly_hours']
cleaned_df['satisfaction_project_interaction'] = cleaned_df['satisfaction_level'] * cleaned_df['number_project']
cleaned_df['project_evaluation'] = cleaned_df['number_project'] * cleaned_df['last_evaluation']
cleaned_df['hours_per_project'] = cleaned_df['average_montly_hours'] / cleaned_df['number_project']
cleaned_df['time_since_last_evaluation'] = cleaned_df['time_spend_company'] - cleaned_df['last_evaluation']
cleaned_df['satisfaction_hours'] = cleaned_df['satisfaction_level'] * cleaned_df['average_montly_hours']
cleaned_df['evaluation_hours'] = cleaned_df['last_evaluation'] * cleaned_df['average_montly_hours']
cleaned_df['work_accident_projects'] = cleaned_df['Work_accident'] * cleaned_df['number_project']
cleaned_df['satisfaction_time_spent'] = cleaned_df['satisfaction_level'] * cleaned_df['time_spend_company']
cleaned_df['evaluation_time_spent'] = cleaned_df['last_evaluation'] * cleaned_df['time_spend_company']
cleaned_df['projects_time_spent'] = cleaned_df['number_project'] * cleaned_df['time_spend_company']
cleaned_df['average_hours_per_year'] = cleaned_df['average_montly_hours'] * 12
cleaned_df['project_satisfaction'] = cleaned_df['number_project'] * cleaned_df['satisfaction_level']
cleaned_df['project_evaluation_diff'] = cleaned_df['number_project'] * cleaned_df['satisfaction_evaluation_diff']
cleaned_df['company_avg_satisfaction'] = cleaned_df.groupby('time_spend_company')['satisfaction_level'].transform('mean')
cleaned_df['company_avg_evaluation'] = cleaned_df.groupby('time_spend_company')['last_evaluation'].transform('mean')
cleaned_df['relative_satisfaction'] = cleaned_df['satisfaction_level'] / cleaned_df['company_avg_satisfaction']
cleaned_df['relative_evaluation'] = cleaned_df['last_evaluation'] / cleaned_df['company_avg_evaluation']
cleaned_df['interaction_1'] = cleaned_df['satisfaction_level'] * cleaned_df['time_spend_company'] * cleaned_df['average_montly_hours']
cleaned_df['interaction_2'] = cleaned_df['last_evaluation'] * cleaned_df['number_project'] * cleaned_df['time_spend_company']

print(cleaned_df.head().T)
print(cleaned_df.shape)
print(cleaned_df.isnull().sum())
imputer = SimpleImputer(strategy='mean')
cleaned_df = pd.DataFrame(imputer.fit_transform(cleaned_df), columns=cleaned_df.columns)
print(cleaned_df.isnull().sum())
cleaned_df.info()

cleaned_df['left'] = cleaned_df['left'].astype(int)

X = cleaned_df.drop('left', axis=1)
y = cleaned_df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000, solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy(Doğruluk):", accuracy * 100)
print("Precision(Kesinlik):", precision * 100)
print("Recall(Duyarlılık):", recall * 100)
print("F1 Score(P - R harmonik ortalamasıdır ve dengeli bir performans ölçüsü sağlar):", f1 * 100)


data = pd.DataFrame({
    'satisfaction_level': [0.5],
    'last_evaluation': [0.6],
    'number_project': [4],
    'average_montly_hours': [200],
    'time_spend_company': [3],
    'Work_accident': [0],
    'promotion_last_5years': [0],
    'Department_IT': [1],
    'Department_RandD': [0],
    'Department_accounting': [0],
    'Department_hr': [0],
    'Department_management': [0],
    'Department_marketing': [0],
    'Department_product_mng': [0],
    'Department_sales': [0],
    'Department_support': [0],
    'Department_technical': [0],
    'salary_high': [0],
    'salary_low': [1],
    'salary_medium': [0],
    'high_work_hours': [0],
    'performance_score': [0.3],
    'work_load': [0.02],
    'satisfaction_evaluation_diff': [0.1],
    'accident_hour_ratio': [0],
    'satisfaction_project_interaction': [2.0],
    'project_evaluation': [400],
    'hours_per_project': [200],
    'time_since_last_evaluation': [0],
    'satisfaction_hours': [0.3],
    'evaluation_hours': [0.6],
    'work_accident_projects': [0],
    'satisfaction_time_spent': [0.3],
    'evaluation_time_spent': [0.6],
    'projects_time_spent': [400],
    'average_hours_per_year': [2400],
    'project_satisfaction': [400],
    'project_evaluation_diff': [400],
    'company_avg_satisfaction': [0.5],
    'company_avg_evaluation': [0.6],
    'relative_satisfaction': [0.5],
    'relative_evaluation': [0.6],
    'interaction_1': [0.5],
    'interaction_2': [0.6]
})

predicted_attrition = model.predict(data)
print("Predicted Attrition (0=No, 1=Yes):", predicted_attrition)
