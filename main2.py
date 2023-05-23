import pandas as pd
from fpgrowth import fpgrowth
from association_rules import association_rules
import matplotlib.pyplot as plt


Hepatitis_df = pd.read_csv('hepatitis.csv')
Lung_cancer_df=pd.read_csv('lung_cancer.csv')

Hepatitis_df['histology'] = (Hepatitis_df['histology'] == 2)
columns = ['sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palable', 'spiders', 'ascites', 'varices']
Hepatitis_df[columns] = (Hepatitis_df[columns] == 2)
Hepatitis_frequent_itemsets = fpgrowth(Hepatitis_df[columns], min_support=0.7, use_colnames=True)
Hepatitis_association_rules = association_rules(Hepatitis_frequent_itemsets, metric="confidence", min_threshold=0.7)
print(Hepatitis_association_rules.to_string())
print(Hepatitis_frequent_itemsets.to_string())


# plt.barh(range(len(Hepatitis_frequent_itemsets)), Hepatitis_frequent_itemsets.support, tick_label=Hepatitis_frequent_itemsets.itemsets)
# plt.xlabel('Support')
# plt.ylabel('Itemsets')
# plt.title('Frequent Itemsets')
# plt.show()

# plt.barh(Hepatitis_association_rules['antecedents'].astype(str) + '->' +  Hepatitis_association_rules['consequents'].astype(str), Hepatitis_association_rules['lift'],color='purple')
# plt.xlabel('Lift')
# plt.ylabel('Association Rule')
# plt.title('Association Rules for Hepatitis Disease Prediction')
# plt.show()

# if 'LUNG_CANCER' in df.columns:
Lung_cancer_df['LUNG_CANCER']=(Lung_cancer_df['LUNG_CANCER']==2)
# Extract the relevant columns for mining
columns =['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC_DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH','SWALLOWING_DIFFICULTY','CHEST_PAIN']
Lung_cancer_df[columns] = (Lung_cancer_df[columns] == 2)
Lung_cancer_frequent_itemsets = fpgrowth(Lung_cancer_df[columns], min_support=0.4, use_colnames=True)
Lung_cancer_association_rules = association_rules(Lung_cancer_frequent_itemsets, metric="confidence", min_threshold=0.7)
# print(Lung_cancer_association_rules.to_string())
# print(Lung_cancer_frequent_itemsets.to_string())

# plt.barh(Lung_cancer_association_rules['antecedents'].astype(str) + '->' +  Lung_cancer_association_rules['consequents'].astype(str), Lung_cancer_association_rules['lift'],color='purple')
# plt.xlabel('Lift')
# plt.ylabel('Association Rule')
# plt.title('Association Rules for Lung Cancer Disease Prediction')
# plt.show()
#
# plt.barh(range(len(Lung_cancer_frequent_itemsets)), Lung_cancer_frequent_itemsets.support, tick_label=Lung_cancer_frequent_itemsets.itemsets)
# plt.xlabel('Support')
# plt.ylabel('Itemsets')
# plt.title('Frequent Itemsets')
# plt.show()

###################################






