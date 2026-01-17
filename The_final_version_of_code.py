import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("projet_python.csv")  # Pour la lecture du fichier csv; c'est à dire DataSet
students = pd.DataFrame(df)

'''print('\n                           Description de dataset                                     \n')

print("Résumé de la dataset:")
print(df.info())

# b) Lister les 10 premières ligne
print("\nLes 10 premieres colonnes:")
print(df.head(10).to_string())

# c) Lister les 10 dernières lignes
print("\nLes 10 dernières colonnes:")
print(df.tail(10).to_string())

# d) Lister toutes les colonnes avec leurs types
print("\nLes colonnes et leurs Types:")
print(df.dtypes)

# e) Lister les valeurs manquantes (colonne vide ou du type NaN-Not a Number)
print("\nLe nombre de valeurs manquantes:")
print(df.isnull().sum())  # compte le nombre par colonne

print('\n                           résumé statistique de dataset                              \n')

print("\nLes indicateurs statistique du dataset:\n")
print(df.describe().to_string())

print('\n                   Nettoyage des données manquantes et les incohérences de type                 \n')

# Dans cette partie, nous allons gérer les valeurs manquantes de notre dataset
# Tout au long de notre programme, nous évoquerons les différents strategies opérées pour avoir un fichier conforme au traitement et a l'analyse.

print("\t\t\t------ REPARTITION DES TYPES DE DONNEES ------")
# Pour un nettoyage optimal, nous constatons que le dataset est forme de colonnes avec deux types de valeurs
# de type numérique : des entiers et des float
# de type chaine de caractères.
# Dans ce cas, nous allons les réunir dans deux listes
col_num = ['id','funding_per_student_usd','avg_test_score_percent','student_teacher_ratio', 'percent_low_income','percent_minority','internet_access_percent','dropout_rate_percent']
col_char= ['school_name','state','school_type','grade_level']

# Nous souhaitons mettre en evidence les dimensions avant et apres nettoyage
# Copions la dataset d'abord
df_nettoyee = df.copy()
print(f"\nDimensions initiales : {df_nettoyee.shape}\n")

print("\t\t\t------ GESTION DES INCOHÉRENCES DE TYPE ------")
# Identification des incohérences dans les colonnes numériques
# On convertit les 'mots' trouvés dans les colonnes numériques en NaN.
for col in col_num:
    nbr_avt = df_nettoyee[col].isnull().sum()
    # Conversion en NaN
    df_nettoyee[col] = pd.to_numeric(df_nettoyee[col], errors='coerce')
    nbr_apres = df_nettoyee[col].isnull().sum()
    incoherences_trouvees = nbr_apres - nbr_avt
    print( f"Colonne '{col}': {incoherences_trouvees} valeurs incorrectes converties en NaN.")
print('\n')
nbre_lignes_avant= len(df_nettoyee)
indices_a_supprimer = set()

# On va convertir les valeurs incorrectes en NaN
# Cela nous permet d'utiliser une simple vérification pd.isna() dans l'étape suivante.
for col in col_char:
    # Supprimons les espaces des chaînes de caractères pour standardiser toutes les lignes
    df_nettoyee[col] = df_nettoyee[col].astype(str).str.strip()

    # Identification et Conversion en NaN des nombres incorrects
    #    Si la valeur est un nombre, elle réussit la conversion (non-NaN).
    valeurs_numeriques_test = pd.to_numeric(df_nettoyee[col], errors='coerce')

    # On remplace les nombres identifiés (non-NaN) par np.nan pour la suppression manuelle
    df_nettoyee.loc[valeurs_numeriques_test.notna(), col] = np.nan

    # 4. On remplace également les chaînes de texte qui sont des manquants par np.nan
    df_nettoyee.loc[df_nettoyee[col].str.lower().isin(['nan', 'none', 'n/a', '']), col] = np.nan
    print(f"On a converti les nombres et chaînes manquantes en NaN.")

# On fait appélle à la condition if pour identifier et collecter les lignes à supprimer (où il y a un NaN)
for index, row in df_nettoyee.iterrows():
    # Parcourir chaque colonne de chaine de caractères
    for col in col_char:
        valeur = row[col]

        # i la valeur est NaN (donc un nombre ou un manquant), ou si elle n'est pas une chaîne de caractères
        # la ligne est invalide.
        if pd.isna(valeur) or not isinstance(valeur, str):
             indices_a_supprimer.add(index)
             break # Une fois qu'on a trouvé un problème, on passe à la ligne suivante

# Apres l'identification on procede a la suppression et on doit renitialiser l'index
df_nettoyee.drop(list(indices_a_supprimer), inplace=True)
df_nettoyee.reset_index(drop=True, inplace=True)

nbre_lignes_supprimees_numerique_cat = nbre_lignes_avant - len(df_nettoyee)
print(f"Total de {nbre_lignes_supprimees_numerique_cat} lignes dans les string ont été supprimées.")
print('\n')
print('\t\t\t------ GESTION DES VALEURS MANQUANTES (SUPPRESSION ET REMPLACEMENT) ------')

# Nous allons supprimer les lignes avec NaN pour les colonnes de chaines de caractères.
# Cela couvre les NaN originaux et toute autre incohérence rendue NaN.
nbre_ligne_avant= len(df_nettoyee)
# Suppression des lignes
df_nettoyee.dropna(subset=col_char, inplace=True)
nbre_ligne_apres = nbre_ligne_avant - len(df_nettoyee)
print(f"Suppression des lignes avec NaN dans Chaînes de Caractères : {nbre_ligne_apres} lignes supprimées.")

# Remplacement des NaN restants dans les colonnes numériques par la valeur mediane.
for col in col_num:
    # Calcul de la mediane des colonnes numériques.
    mediane = df_nettoyee[col].median()
    # Remplacement des NaN par la médiane de la colonne
    nbre_nan= df_nettoyee[col].isnull().sum()
    # Remplacement des NaN
    df_nettoyee[col] = df_nettoyee[col].fillna(mediane)
    print(f"Colonne '{col}': {nbre_nan} valeurs remplacées par la Médiane ({mediane:.2f}).")
print('\n')
print("\t\t\t------ GESTION DES LIGNES DUPLIQUÉES ------")
# Trouvons les lignes dupliquees.
lignes_dupliquees = df_nettoyee.duplicated().sum()
if lignes_dupliquees > 0:
    # Supprimons les lignes dupliquées et mettons à jour la dataset
    df_nettoyee.drop_duplicates(inplace=True)
    print(f"{lignes_dupliquees} lignes dupliquées ont été supprimées.")
else:
    print("Aucune ligne dupliquée trouvée.")
print('\n')
print('\t\t\t------ RÉSUMÉ DU NETTOYAGE ------')
print(f"Dimensions initiales : {students.shape[0]} lignes.")
print(f"Dimensions après nettoyage : {df_nettoyee.shape[0]} lignes.")
print(f"Total de lignes supprimées (Typage/Manquantes/Doublons) : {df.shape[0] - df_nettoyee.shape[0]}")
print('\n')
# A present vérifions que la dataset a ete bien nettoyee
print("Vérification finale des valeurs manquantes:")
# On doit avoir un affichage de 0 pour chaque colonne car on calcule la somme des NaN pour les colonnes traitées
print(df_nettoyee[col_num + col_char].isnull().sum())
print('\t\t\t------ SAUVEGARDE DU DATASET CORRIGÉ ------')
projet_python_propre = 'projet_python.csv'
df_nettoyee.to_csv(projet_python_propre, index=False)

print(f"La dataset nettoyée a été sauvegardée dans 'projet_python.csv'.")

print('\n                     Groupement et filtration du dataset                               \n')

print('\n                                     Filtration                                     \n')
# filtering by state to see the total number of schools in NY that have an average greater than 80
print("The schools in New York with an average test score greater than 70:")
NY_Schools = students[(students['state'] == 'New York') & (students['avg_test_score_percent'] > 80)]
# We'll add the head() instruction for showing just the five Rows of the filtering dataframe
print(NY_Schools[['school_name', 'avg_test_score_percent']].head())
# we turn it to csv file to see the complete dataframe
NY_Schools.to_csv('NY_Schools.csv')

print('----------------------------------------------------------')

# Filtering by budget to see the schools that are affording a budget for their students that's greater than 20000$
rich_schools = students[students['funding_per_student_usd'] > 21000]
print("Schools with a big budget:")
# We'll add the head() instruction for showing just the five Rows of the filtering dataframe
print(rich_schools[['school_name','state', 'funding_per_student_usd']].head())
rich_schools.to_csv('rich_schools.csv')

print('----------------------------------------------------------')

# Filtering by student_teacher_ratio to see how many schools have less than 15 students per class
small_classes = students[students['student_teacher_ratio'] < 15]
print("Schools with a ration less than 15:")
print(small_classes[['school_name','state', 'student_teacher_ratio']].head())
small_classes.to_csv('small_classes.csv')

print('----------------------------------------------------------')

# Grouping Instructions

print('\n                                     Grouping                                     \n')

# Calculating the schools number for each state
print("Number of schools by state:")
Initial_schools_number = students.groupby('state').size()
schools_number = Initial_schools_number.sort_values(ascending=False)
print(schools_number)
schools_number.to_csv('schools_Number.csv')

print('----------------------------------------------------------')

# Grouping schools by their type and calculating the mean of the these numerical columns: avg_test_score_percent et funding_per_student_usd
print("Mean of schools by their type:")
schools_mean = students.groupby('school_type')[['avg_test_score_percent', 'funding_per_student_usd']].mean()
schools_mean = schools_mean.sort_values(by='avg_test_score_percent', ascending=False)
print(schools_mean)
schools_mean.to_csv('schools_Mean.csv')

print('----------------------------------------------------------')

# Grouping schools by their type and calculating the average of avg_test_score_percent for each state, and doing a sorting to see the first states in the grade
print("The classification of the states based on the average of their schools:")
state_scores = students.groupby('state')['avg_test_score_percent'].mean()
state_scores = state_scores.sort_values(ascending=False)
print(state_scores)
state_scores.to_csv('schools_Mean.csv')'''

print('\n                                Visualization Tasks                                 \n')

#-------------------------------------SCATTER---------------------------------#
# A. Relation entre Financement et Performance, codée par le type d'école
'''plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='funding_per_student_usd',
    y='avg_test_score_percent',
    hue='school_type', # Utilise la couleur pour distinguer Public/Private
    data=df,
    alpha=0.6, # Transparence pour mieux voir la densité des points (1000 points)
    s=50 # Taille des points
)
plt.title("Relation entre Financement par Élève et Score Moyen aux Tests")
plt.xlabel("Financement par Élève (USD)")
plt.ylabel("Score Moyen aux Tests (%)")
plt.grid(axis='y', linestyle='--')
plt.show()'''

#-------------------------------------BARPLOT---------------------------------#
'''data1=df.groupby('school_type')['percent_low_income'].sum()
plt.bar(data1.index,data1,color='brown',width=0.7)

# Titre
plt.title('Cumul du Taux de Revenu Faible par Catégorie d\'École',color = 'red',fontweight ='bold',fontsize=14)

# Axe X
plt.xlabel('Catégorie d\'École', color='green',fontsize=12)

# Axe Y
plt.ylabel('Taux de Faible Revenu (%)',color='green',fontsize=12)

# Grille
plt.grid(True,color = 'gray',linestyle = '--',linewidth=0.4,alpha=0.8)
plt.show()'''
#-------------------------------------HISTOGRAMME---------------------------------#
# Histogramme de visualisation du score

'''plt.figure(figsize=(10, 6))
plt.hist(
    df['avg_test_score_percent'],
    bins=25,
    edgecolor='black', # Bordures pour distinguer les barres
    color='skyblue'
)
plt.title("Distribution du score par Élève",color='skyblue',fontsize=20,fontweight='bold')
plt.xlabel("Score par Élève",fontweight='bold')
plt.ylabel("Fréquence (Nombre d'Écoles)",fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()'''

# 2.1-----------------------------------SCHOOL BOX PLOT--------------------------------------------#
# 1. Préparation des Données : Créer la liste des listes
'''types_ecole = df['school_type'].unique()
listes_scores = []
for type_level in types_ecole:
    # Ajoute la série des scores pour ce type d'école à la liste principale
    listes_scores.append(df[df['school_type'] == type_level]['avg_test_score_percent'])

plt.figure(figsize=(10, 7))

# 2. Utilisation de plt.boxplot
box1 = plt.boxplot(
    listes_scores,
    labels=types_ecole,  # Étiquettes pour l'axe X
    patch_artist=True,  # Important pour pouvoir colorier les boîtes
    widths=0.6  # Largeur des boîtes
)

# 3. Personnalisation des Couleurs
couleurs_boites = ['salmon', 'dodgerblue', 'mediumseagreen']

for patch, color in zip(box1['boxes'], couleurs_boites):
    patch.set_facecolor(color)

plt.title("Distribution du Score Moyen aux Tests par ecole",color='steelblue',fontweight='bold',fontsize=20)
plt.xlabel("Score par ecole",fontweight='bold')
plt.legend(types_ecole)
plt.ylabel("Score Moyen aux Tests (%)",fontweight='bold')
plt.grid(axis='y', alpha=0.5)
plt.show()'''

#-------------------------------------PIE PLOT---------------------------------#
'''school_counts = df['school_type'].value_counts()
mycolors = ["gray","yellow","indigo"]
explode = [0.1,0.1,0.1]
plt.pie(school_counts,labels=school_counts.index,autopct='%1.1f%%',startangle=90,
        colors=mycolors,textprops={'fontsize': 12, 'fontweight': 'bold'},explode=explode,
        shadow=True)
plt.title("Répartition des types d’écoles",fontsize = 15, fontweight = 'bold')
plt.legend(title = "Types :")
plt.tight_layout()
plt.show()'''

print('\n                                     OOP Implementation:                                     \n')

class EducationAnalysis:
     ####################### LOAD DATA ########################################
     #################################### Initialisation du chemin ##################################
    def __init__(self, chemin_fichier=None):
            self.chemin_fichier = chemin_fichier
            self.df = None

        #######################################Charger la data ###########################################
    def load_data(self):
        self.df = pd.read_csv(self.chemin_fichier)
        ################################ load plot ##########################################
    def plot_data(self):
         if self.df is None:
             print("Data pas charger. Cannot plot.")
             return
         # Relation entre Financement et Performance, codée par le type d'école
         '''plt.figure(figsize=(10, 6))
         sns.scatterplot(
             x='funding_per_student_usd',
             y='avg_test_score_percent',
             hue='school_type',  # Utilise la couleur pour distinguer Public/Private
             data=df,
             alpha=0.6,  # Transparence pour mieux voir la densité des points (1000 points)
             s=50  # Taille des points
         )
         plt.title("Relation entre Financement par Élève et Score Moyen aux Tests")
         plt.xlabel("Financement par Élève (USD)")
         plt.ylabel("Score Moyen aux Tests (%)")
         plt.grid(axis='y', linestyle='--')
         plt.show()'''

         # Histogramme de visualisation du score

         '''plt.figure(figsize=(10, 6))
         plt.hist(
             df['avg_test_score_percent'],
             bins=25,
             edgecolor='black', # Bordures pour distinguer les barres
             color='skyblue'
         )
         plt.title("Distribution du score par Élève",color='skyblue',fontsize=20,fontweight='bold')
         plt.xlabel("Score par Élève",fontweight='bold')
         plt.ylabel("Fréquence (Nombre d'Écoles)",fontweight='bold')
         plt.grid(axis='y', linestyle='--', alpha=0.5)
         plt.show()'''

         # 2.1-----------------------------------SCHOOL--------------------------------------------#
         # 1. Préparation des Données : Créer la liste des listes
         '''types_ecole = df['school_type'].unique()
         listes_scores = []
         for type_level in types_ecole:
             # Ajoute la série des scores pour ce type d'école à la liste principale
             listes_scores.append(df[df['school_type'] == type_level]['avg_test_score_percent'])

         plt.figure(figsize=(10, 7))

         # 2. Utilisation de plt.boxplot
         box1 = plt.boxplot(
             listes_scores,
             labels=types_ecole,  # Étiquettes pour l'axe X
             patch_artist=True,  # Important pour pouvoir colorier les boîtes
             widths=0.6  # Largeur des boîtes
         )

         # 3. Personnalisation des Couleurs
         couleurs_boites = ['salmon', 'dodgerblue', 'mediumseagreen']

         for patch, color in zip(box1['boxes'], couleurs_boites):
             patch.set_facecolor(color)

         plt.title("Distribution du Score Moyen aux Tests par ecole",color='steelblue',fontweight='bold',fontsize=20)
         plt.xlabel("Score par ecole",fontweight='bold')
         plt.legend(types_ecole)
         plt.ylabel("Score Moyen aux Tests (%)",fontweight='bold')
         plt.grid(axis='y', alpha=0.5)
         plt.show()'''

         # -------------------------------------PIE PLOT---------------------------------#
         '''school_counts = df['school_type'].value_counts()
         mycolors = ["gray","yellow","indigo"]
         explode = [0.1,0.1,0.1]
         plt.pie(school_counts,labels=school_counts.index,autopct='%1.1f%%',startangle=90,
                 colors=mycolors,textprops={'fontsize': 12, 'fontweight': 'bold'},explode=explode,
                 shadow=True)
         plt.title("Répartition des types d’écoles",fontsize = 15, fontweight = 'bold')
         plt.legend(title = "Types :")
         plt.tight_layout()
         plt.show()'''

     ##################################analyse load #############################################
    def analyze_data(self):
        """
        Analyse descriptive de base du dataset
        """
        print("=== 5 premières lignes ===")
        print(self.df.head().to_string())

        print("--- 5 dernières lignes ---")
        print(self.df.tail().to_string())

        print("--- Les noms des colonnes and leurs types ---")
        print(self.df.dtypes)

        print("--- Valeurs Manquantes par colonne---")
        print(self.df.isna().sum())

        print("--- Statistique de base ----")
        print(self.df.describe().to_string())
    ################################## Application #############################################
donnees = EducationAnalysis("projet_python.csv")
donnees.load_data()
donnees.analyze_data()
donnees.plot_data()
