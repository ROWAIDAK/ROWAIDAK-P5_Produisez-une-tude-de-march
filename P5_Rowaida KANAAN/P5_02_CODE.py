#!/usr/bin/env python
# coding: utf-8

# # Produisez une étude de marché
# **Scénario**
# # Mission 1: Construisez l'échantillon contenant l'ensemble des pays disponibles.
#    1. La population par pays
#    2. Données sur les bilans alimentaires mondiaux (2019)
#    3. Les valeurs de PIB par habitant,
#    4. Production viande de Volailles, taux d'autosuffisanc
#    5. Dataframe principal   
# # Mission 2 : réalisation d'un dendrogramme
# 1. Environnement
# 2. Aperçu des corrélations
# 3. Classification des pays via Clustering Hiérarchique Ascendant (CHA)
# 4. Attribution des 5 groupes et World map de répartition
# 5. Centroïdes des clusters
# 6. Description et critique des clusters
# 
# # Mission 3 : Analyse en Composantes Principales (ACP)
# 1. Application de l'algorithme du K-Means
# 2. Visualisation des clusters en ACP pour la projection des données
# 3. ACP - Cercle des corrélations¶
# 4. Sélections des pays sur groupes Kmeans.
# # Mission 4 : Tests statistiques
# 3. Test d'adéquation de Kolmogorov-Smirno
# 2. Tester l'égalité de la variance
# 
# Le projet est sur github
# 
# https://github.com/ROWAIDAK/ROWAIDAK-P5_Produisez-une-tude-de-march

# # Scénario
# Votre entreprise **d'agroalimentaire** souhaite se développer à l'international. Elle est spécialisée dans...
#  le poulet !
# 
# L'international, oui, mais pour l'instant, le champ des possibles est bien large : aucun pays particulier ni aucun continent n'est pour le moment choisi. Tous les pays sont envisageables !
# 
# Votre objectif sera d'aider **à cibler plus particulièrement certains pays**, dans le but d'approfondir ensuite l'étude de marché. Plus particulièrement, l'idéal serait de produire des "groupes" de pays, plus ou moins gros, dont on connaît les caractéristiques.
# 
# Dans un premier temps, la stratégie est plutôt d'exporter les produits plutôt que de produire sur place, c'est-à-dire dans le(s) nouveau(x) pays ciblé(s).

# 
# Pour identifier les pays propices à une insertion dans le marché du poulet,
#  Il vous a été demandé de cibler les pays. 
#  Etudier les régimes alimentaires de chaque pays, notamment en termes de protéines d'origine animale et en termes de calories.
# 
# 
# Construisez votre échantillon contenant l'ensemble des pays disponibles, chacun caractérisé par ces variables :
# 
# différence de population entre une année antérieure (au choix) et l'année courante, exprimée en pourcentage ;
# proportion de protéines d'origine animale par rapport à la quantité totale de protéines dans la disponibilité alimentaire du pays ;
# disponibilité alimentaire en protéines par habitant ;
# disponibilité alimentaire en calories par habitant.
# 
# Construisez un dendrogramme contenant l'ensemble des pays étudiés, puis coupez-le afin d'obtenir 5 groupes.
# 
# Caractérisez chacun de ces groupes selon les variables cités précédemment, et facultativement selon d'autres variables que vous jugerez pertinentes (ex : le PIB par habitant). Vous pouvez le faire en calculant la position des centroïdes de chacun des groupes, puis en les commentant et en les critiquant au vu de vos objectifs.
# 
# Donnez une courte liste de pays à cibler, en présentant leurs caractéristiques. Un découpage plus précis qu'en 5 groupes peut si besoin être effectué pour cibler un nombre raisonnable de pays. 
# 
# Visualisez vos  partitions dans le premier plan factoriel obtenu par ACP.
# 
# Dans votre partition, vous avez obtenu des groupes distincts. Vérifiez donc qu'ils diffèrent réellement. Pour cela, réalisez les tests statistiques suivants :
# 
# un test d'adéquation : parmi les 4 variables, ou parmi d'autres variables que vous trouverez pertinentes, trouvez une variable dont la loi est normale ;
# un test de comparaison de deux populations (dans le cas gaussien) : choisissez 2 clusters parmi ceux que vous aurez déterminé. Sur ces 2 clusters, testez la variable gaussienne grâce à un test de comparaison.

# In[374]:


# Librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import sklearn as sk
import scipy as sp
import pca as pca
from sklearn import cluster
from sklearn.cluster import KMeans


# In[375]:


#Versions utilisées
print("Jupyter Notebook : " + str(pd.__version__))
print("Pandas : " + str(pd.__version__))
print("Numpy : " + str(np.__version__))
print("Seaborn : " + str(sns.__version__))

# Styles Seaborn
sns.set( 
    style='whitegrid',
    context='notebook',
    palette='Paired',
    rc={'figure.figsize':(8,5)})


# # Mission 1:  Construisez l'échantillon contenant l'ensemble des pays disponibles.
# 

# ## 1. La population par pays
# la variation de la population entre 2009 et 2019, en %, qui sera positive en cas de croissance ou négative en cas de baisse démographique ;

# Sources FAO (https://www.fao.org/faostat/fr/#data/OA)

# In[376]:


df_population = pd.read_csv('datas/fao-populations_2019- 2009.csv' , header=0, sep=",", decimal=".")
df_population.head(2)


# In[377]:


df_population = df_population[['Code zone (FAO)', 'Zone', 'Année', 'Valeur']]
df_population['Valeur'] = df_population['Valeur']*1000
#pivot
df_population = df_population.pivot_table(index=['Code zone (FAO)','Zone'], columns='Année', values = 'Valeur', aggfunc = sum).reset_index()
#rename
df_population.columns = ['Code zone (FAO)','Zone', '2009', '2019']

#variable creation
df_population['Évolution population (%)'] = round((df_population['2019'] - df_population['2009']) /
                                               df_population['2019'] *100,2)

df_population= df_population.rename(columns= {'2019': 'population',})
df_population= df_population[[ 'Code zone (FAO)','Zone'  ,'population' ,'Évolution population (%)']]

#show

df_population.head(2)


# ## 2. Données sur les bilans alimentaires mondiaux (2019)
# 

# 
# Sources FAO (http://www.fao.org/faostat/fr/#data)
# 

# In[378]:


df_dispo_alimentaire = pd.read_csv('datas/les bilans alimentaires 2019 2.csv', header=0, sep=",", decimal=".")
#Il y a un problème avec le site Web de la FAO traitant de la langue française lors du téléchargement du fichier
df_dispo_alimentaire.head(3)


# In[379]:


df_dispo_alimentaire = df_dispo_alimentaire.pivot_table(index=[ 'Zone','Code zone (FAO)'],
                                columns=['?l?ment', 'Produit'],
                                values = 'Valeur',
                                aggfunc=sum).reset_index()
df_dispo_alimentaire.head()


# In[380]:


df_dispo_alimentaire["ratio_proteines_animales(%)"] = round((df_dispo_alimentaire[('Disponibilit? de prot?ines en quantit? (g/personne/jour)', 'Produits Animaux')]
                                                                         /df_dispo_alimentaire[('Disponibilit? de prot?ines en quantit? (g/personne/jour)', 'Total General')])*100,2)
df_dispo_alimentaire.head(2)


# In[381]:


df= pd.merge(df_population, df_dispo_alimentaire, on="Code zone (FAO)")

df.head()


# In[382]:


df = df[['Code zone (FAO)','Zone', 'population','Évolution population (%)',
         ('Disponibilit? alimentaire (Kcal/personne/jour)', 'Total General'),
         ('Disponibilit? de prot?ines en quantit? (g/personne/jour)', 'Total General'),
         ('ratio_proteines_animales(%)','') ]]
df.head()


# In[383]:


df.rename(columns={('Disponibilit? alimentaire (Kcal/personne/jour)', 'Total General'):'dispo_calories',
                              ('Disponibilit? de prot?ines en quantit? (g/personne/jour)', 'Total General'):'dispo_proteines',
                            ('ratio_proteines_animales(%)','' ):'ratio_prot_anim'}
                     ,inplace=True)

df['dispo_calories'] = df['dispo_calories']*365
df['dispo_proteines'] = df['dispo_proteines']*365
df.head()


# ## 3.  Les valeurs de PIB par habitant,
# https://www.fao.org/faostat/fr/#data/FS

# In[384]:


# Dataframes des données complémentaires
# Indicateurs Macro (PIB et croissance)
df_PIB_habitant = pd.read_csv('datas/PIB.csv', header=0, sep=',', decimal='.')

df_PIB_habitant.head(2)


# In[385]:


df_PIB_habitant = df_PIB_habitant[['Code zone (FAO)','Élément','Valeur','Produit']]
                        

df_PIB_habitant = df_PIB_habitant.pivot_table(index=['Code zone (FAO)'], columns='Élément', values='Valeur', aggfunc=sum).reset_index()
df_PIB_habitant = df_PIB_habitant.rename(columns={'Valeur US $ par habitant':"PIB_par_habitant" })
df_PIB_habitant = df_PIB_habitant[['Code zone (FAO)','PIB_par_habitant']]

df_PIB_habitant.head(2)


# ## 4. Production viande de Volailles, taux d'autosuffisance
# défini en économie comme le rapport entre les importations et la disponibilité intérieure du pays ;
# 
# https://www.fao.org/faostat/fr/#data/FBS

# In[386]:


df_viande_volailles = pd.read_csv('datas/la viande de volailles1!.csv', header=0, sep=",", decimal=".")
df_viande_volailles.head(2)


# In[387]:


df_viande_volailles = df_viande_volailles.pivot_table(index=['Code zone (FAO)'],
                                columns=['Élément'],
                                values = 'Valeur',
                                aggfunc=sum).reset_index()
#Le taux_suffisance= Production  ÷ (Production alimentaire domestique + importations ー exportations) ×100
df_viande_volailles['taux_suffisance']=((df_viande_volailles['Production'] ) / 
                                           (df_viande_volailles['Disponibilité intérieure']))*100



#le taux de dépendance aux importations, défini en économie comme le rapport entre les importations et la disponibilité intérieure du pays ;

df_viande_volailles['dep_import']=(df_viande_volailles['Importations - Quantité'] /
                                                        df_viande_volailles['Disponibilité intérieure'])*100


#Le taux d'auto-suffisance alimentaire est un indice permettant de mesurer l'importance de la production alimentaire d'un pays par rapport à sa consommation intérieure.
df_viande_volailles = df_viande_volailles[['Code zone (FAO)','taux_suffisance','dep_import']]

df_viande_volailles.head(2)


# **Le taux d'auto-suffisance alimentaire** est un indice permettant de mesurer l'importance de la production alimentaire d'un pays par rapport à sa consommation intérieure.
# 
# 
# 

# # Dataframe principal

# In[388]:


df= pd.merge(df, df_PIB_habitant, on="Code zone (FAO)")


df= pd.merge(df, df_viande_volailles, on="Code zone (FAO)")
df.head(2)


# In[389]:


df.info()


# In[390]:


#Trouver les valeurs manquantes
df_null=  df.loc[df.isnull().any(axis=1)]
df_null


# In[391]:


df=df.dropna()


# In[392]:


df.describe()


# In[393]:


#Retrait du pays 'France' sur notre échantillon car nous n'exportons pas vers notre pays. 

df = df[df['Zone'] != 'France']


# In[394]:


#Calcul de fréquence en Chine
df[df['Zone'].str.contains('hin')]


# In[395]:


#Supprimer la ligne Chine  car elle est en double. 
df = df[df['Zone'] != 'Chine']
### Suppression de la corée du Nord
df = df[df['Zone'] != 'République populaire démocratique de Corée']


# In[396]:


#Vérification d'éventuelles valeurs manquantes et/ou en doubles dans l'échantillon
print(df.duplicated().sum())
print(df.isna().sum())


# In[397]:


### Suppression des petits pays
df = df[df.population >= 500000]


# In[398]:


df= df.rename(columns={'Zone': 'pays'} )


# In[399]:


df.style.background_gradient(cmap='BrBG')


# In[400]:


df.to_csv('exports/df.csv', index=False)
df.shape


# # Mission 2 : réalisation d'un dendrogramme
# 
# 
# Construisez un dendrogramme contenant l'ensemble des pays étudiés, puis coupez-le afin d'obtenir 5 groupes.
# 
# Caractérisez chacun de ces groupes selon les variables cités précédemment, et facultativement selon d'autres variables que vous jugerez pertinentes (ex : le PIB par habitant). Vous pouvez le faire en calculant la position des centroïdes de chacun des groupes, puis en les commentant et en les critiquant au vu de vos objectifs.
# 
# Donnez une courte liste de pays à cibler, en présentant leurs caractéristiques. Un découpage plus précis qu'en 5 groupes peut si besoin être effectué pour cibler un nombre raisonnable de pays. 
# 

# ## Environnement
# 

# In[401]:


#pays comme index
df_clus=df.set_index('pays', drop=True, append=False, inplace=False, verify_integrity=False)

# préparation des données pour le clustering
df_clus = df_clus[["Évolution population (%)", "dispo_calories",
                                 "dispo_proteines",'ratio_prot_anim',
                                 'PIB_par_habitant','taux_suffisance','dep_import']]

df_clus.head(2)


# ## Aperçu des corrélations

# In[402]:


plt.figure(figsize=(15,5))

mask = np.zeros_like(df_clus.corr())
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df_clus.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')

plt.xticks(rotation=25, ha='right')
plt.title('La corrélation des variables',  fontsize=18, pad=20)
plt.savefig('exports/La corrélation des variables.')
plt.show()


# #### Observations
# on peut vérifier la corrélation des variables avec une matrice des corrélations. 
# On peut constater une corrélation positive forte entre  la dispo en protéines et la dispo en calories , le ratio de protéines animales, le PIB.
# Dans une moindre mesure, on retrouve également une corrélation négative entre le pourcentage d'évolution de la population et les différents régimes alimentaires des pays. 
# On note une relation négative entre l'autosuffisance et le pourcentage de dépendance aux importations
# 
# **Nous sommes intéressés par les pays les plus susceptibles de consommer du poulet, donc ceux ayant un fort ratio de protéines animales. Les corrélations montrent que ces pays sont susceptibles d'avoir un fort PIB, et de grandes disponibilités en protéines et calories. 
# Nous nous intéressons également aux pays dépendants des importations qui ne sont pas autosuffisants.**

# ## Classification des pays via Clustering Hiérarchique Ascendant (CHA)
# 
# La classification sera établie sur la base des variables suivantes :
# 
# **La différence de population entre l'année 2013 et l'année 2019, exprimée en pourcentage;\
# La proportion de protéines d'origine animale par rapport à la quantité totale de protéines dans la disponibilité alimentaire du pays;\
# La disponibilité alimentaire en grammes de protéines par habitant ;\
# La disponibilité alimentaire en Kcal par habitant.\
# PIB par habitant.\
# Rapport de dépendance à l'importation.\
# La taux suffisance(٪).**
# 
# L'échantillon comporte peu de variables sur la dimension dite du "Régime alimentaire" et et variables économiques, il comporte également un nombre de pays "maîtrisables" qui permet de commencer par une classification hiérarchique. Algorithme qui a une forte complexité algorithmique en temps et en espace, le clustering hiérarchique est recommandé pour les petits échantillons.
# 
# 
# 
# Le clustering permet de regrouper des individus similaires, c'est-à-dire qu'il va partitionner l'ensemble des individus. On cherche donc à ce que les groupes soient :
# 
#     **Resserrés sur eux-mêmes : deux points qui sont proches devraient appartenir au même groupe.
#   
#     **Loin les uns des autres, c’est-à-dire qu’ils soient fortement différenciés.
# 
# Au préalable, il est nécessaire de centrer-réduire les données. C’est à dire, recalculer chaque valeur de manière à ce que la moyenne de la variable soit égale à 0 et la variance et l’écart-type égalent 1. Pour une variable donnée, on soustrait à chaque valeur la moyenne de la variable, puis on divise le tout par l’écart-type.
# 
# Ensuite, nous pouvons procéder à la classification ascendante hiérarchique selon la méthode de Ward. 
# 

# In[403]:



from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


# dans array numpy
X = df_clus.values 
#nous allons extraire les données d'expression de 153 pays  dans un tableau de données numériques .
 #X comporte uniquement les données  , il ne contient pas les étiquettes des échantillons.

pays = df_clus.index
 #Conservons les étiquettes de chaque échantillon  dans la variable pays.

#centering and reduction
#il est nécessaire de centrer-réduire les données. C’est à dire, recalculer chaque valeur de manière à ce que 
#la moyenne de la variable soit égale à 0 et la variance et l’écart-type égalent 1. Pour une variable donnée,
#on soustrait à chaque valeur la moyenne de la variable, puis on divise le tout par l’écart-type.


std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
#Standardisez les caractéristiques en supprimant la moyenne et  en divisant par l'écart type

#X_scaled =array([[-0.90652015,  0.30068128,  0.61600031, ..., -0.47817227,
        #-1.15900796,  0.84483273],
       #[ 1.29355645, -1.35075836, -1.15709131, ..., -0.68784567,
        #-0.7198978 ,  0.54941119],...,

        

# Clustering hiérarchique avec librairie scipy
z=linkage(X_scaled, method='ward' ,metric='euclidean')
#euclidien pour mesurer la distance entre les observations au sein d'une classe
#ward pour mesurer la distance entre les rangées

# Affichage du dendogramme
plt.figure(figsize=(12,25), dpi=300)
plt.title('Dendogramme de classification ascendante hiérarchique (CAH)')
plt.xlabel('distance')
dendrogram(
    z,#Regroupement hiérarchique encodé avec la matrice renvoyée par la fonction de linkage.
    labels = pays,
    orientation = "right",
    color_threshold=7
)
plt.savefig('exports/dendogram_CAH_1.png')
plt.show()


# Une fois le dendrogramme obtenu, nous pouvons choisir en combien de clusters nous pouvons diviser notre arbre. Ici, une partition en 5 clusters permet d’avoir des groupes de taille raisonnable à analyser.
# 
# Graphiquement, on voit bien que la méthode de Ward a permis de minimiser les distances intra-classes à chaque regroupement.
# 
# 
# Essayons de caractériser chacun de ces clusters en regardant la valeur de leurs centroïde pour chaque variable. Les centroïdes représentent tout simplement la valeur moyenne d’une variable pour un cluster donné.
# Différences entre les clusters :

# In[404]:


plt.figure(figsize=(12,8), dpi=300)
plt.title('Dendogramme de classification ascendante hiérarchique tronqué')
plt.xlabel('distance')
plt.grid(False)
dendrogram(
    z,      #linkage(X_cr, method='ward' ,metric='euclidean'),
    truncate_mode='lastp', #Les p derniers clusters non singleton formés dans la liaison sont les seuls nœuds non feuilles dans la liaison ; 
    p = 5,
    labels = pays,
    orientation = "right",
    show_contracted=True,)
plt.savefig('exports/dendogram_CAH_truncated_1.png')
plt.show()


# ## Attribution des 5 groupes et World map de répartition

# In[405]:


# Coupage du dendrogramme en 5 clusters avec Scipy
groupe_cah = fcluster(z, 5, criterion='maxclust')
#array([2, 1, 4, 5, 2, 2, 5, 3, 3, 2, 4,....
#fcluster: formez des clusters plats à partir du clustering hiérarchique défini par la matrice de liaison donnée.

#affichage des pays et leurs groupes
df_groupage_1 = pd.DataFrame(columns=["groupe_cah","pays"])
df_groupage_1["pays"] = df_clus.index
df_groupage_1["groupe_cah"] = groupe_cah

df_groupage_1


# In[406]:


# Jointure pour ajout des groupes dans le dataframe principal
df_groupes_cah = pd.merge(df_clus, df_groupage_1, on="pays")
df_groupes_cah.to_csv('exports/groupes_cah.csv', index=False)
df_groupes_cah.sample(5)


# ## Centroïdes des clusters

# In[410]:


#Première comparaison des moyennes afin d'identifier le groupe de pays le plus porteur à ce niveau de l'analyse
centroïdes_CAH=df_groupes_cah.groupby('groupe_cah').mean()
centroïdes_CAH.to_csv('exports/P5 centroïdes CAH.csv', index=False)
centroïdes_CAH


# In[285]:


#Préparation de sous-ensembles permettant de caractériser les groupes un à un
df_groupe1_cah = df_groupes_cah[df_groupes_cah['groupe_cah'] == 1]
df_groupe2_cah = df_groupes_cah[df_groupes_cah['groupe_cah'] == 2]
df_groupe3_cah = df_groupes_cah[df_groupes_cah['groupe_cah'] == 3]
df_groupe4_cah = df_groupes_cah[df_groupes_cah['groupe_cah'] == 4]
df_groupe5_cah = df_groupes_cah[df_groupes_cah['groupe_cah'] == 5]


# In[286]:




#Pays du groupe 1 et 2 identifiés comme potentiellement intéressants

print('groupe 1')
print('----------')
print(df_groupe1_cah['pays'].unique())
print('------------------------------------------------------------')
print('------------------------------------------------------------')


print('groupe 2')
print('----------')
print(df_groupe2_cah['pays'].unique())
print('------------------------------------------------------------')
print('------------------------------------------------------------')


print('groupe 3')
print('----------')
print(df_groupe3_cah['pays'].unique())
print('------------------------------------------------------------')
print('------------------------------------------------------------')


print('groupe 4')
print('----------')
print(df_groupe4_cah['pays'].unique())
print('------------------------------------------------------------')
print('------------------------------------------------------------')


print('groupe 5')
print('----------')
print(df_groupe5_cah['pays'].unique())
print('------------------------------------------------------------')
print('------------------------------------------------------------')





# ## Description et critique des clusters

# In[287]:




#Comparaison visuelle des groupes par Boxplot, en abscisse les numéros des groupes
plt.figure(figsize=(20, 20))
sns.set(style="whitegrid")

plt.subplot(221)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='dispo_calories')


plt.subplot(222)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='dispo_proteines')

plt.subplot(223)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='ratio_prot_anim')

plt.subplot(224)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='Évolution population (%)')


plt.savefig("exports/boxplot_dendogramme3.png")




plt.show(block=False)


# In[288]:


#Comparaison visuelle des groupes par Boxplot, en abscisse les numéros des groupes
plt.figure(figsize=(20, 20))
sns.set(style="whitegrid")

plt.subplot(221)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='PIB_par_habitant')


plt.subplot(222)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='dep_import')


plt.subplot(223)
sns.boxplot(data=df_groupes_cah, x='groupe_cah', y='taux_suffisance')


plt.savefig("exports/boxplot_dendogramme3.png")




plt.show(block=False)


# Description et critique des clusters 
# 
# Les groupes 1 et 2, situés principalement en Afrique, ont la démographie la plus forte et le PIB par habitant le plus bas. 
# 
# Les groupes 4 et 5, au contraire, reflètent des pays plus riches en moyenne, comme les États-Unis et la plupart des pays européens. 
# 
# Fait intéressant, le groupe 5 a un bon PIB et une forte croissance démographique.
# 
# En termes de disponibilité alimentaire, ces tendances suivent les mêmes tendances que le PIB. Les pays les plus pauvres ont moins accès à la nourriture. 
# 
# La consommation de protéines animales confirme le partage, avec une forte alimentation carnée de la part des groupes 3,4 et 5 
# On voit que les groupes 2 et 5 importent beaucoup plus de viande de volaille qu'ils n'en produisent. 
# 
# En fait, ils ont les taux d'autosuffisance les plus bas. Au contraire, les groupes 2, 3 et 4 sont raisonnablement autosuffisants. 
# 
#  
# Compte tenu de tous ces critères, quel groupe serait apte à être sélectionné comme candidat pour notre marché international ? 
# 
#  Je pense que les groupes qui dépendent fortement des importations sont les meilleures cibles pour nos ventes.
# Nous choisirons le groupe 5 car il est fortement dépendant des importations, a un taux de croissance démographique élevé et un PIB par habitant élevé.
# 
# 
# Nous choisissons également le groupe 4, car le PIB par habitant est beaucoup plus élevé dans ces pays, ce qui nous permettra de vendre notre production plus facilement et à un meilleur prix. Et le plus élevé en termes de consommation de protéines animales et de calories. Il comprend également des pays géographiquement proches de la France.
#  

# In[289]:


df_select_pays_cah = df_groupes_cah[df_groupes_cah["groupe_cah"].isin([4,5]) == True]

df_select_pays_cah.to_csv('exports/Sélections des pays sur CAH.csv', index=False)
df_select_pays_cah.shape
df_select_pays_cah


# In[290]:


df_select_pays_cah.shape


# En conclusion pour cette première section, 26 pays sont susceptibles de devenir une cible appropriée pour l'entreprise. La demande sera présente dans ces pays. Appliquons une autre méthode, la méthode K-Means, afin de pouvoir comparer cette première sélection.

# # Mission 3 : Analyse en Composantes Principales (ACP)
# 

# 
# Le clustering K-Means est une méthode de clustering simple mais puissante qui crée 𝑘 segments distincts des données où la variation au sein des clusters est aussi petite que possible. Pour trouver le nombre optimal de clusters, je vais essayer différentes valeurs de 𝑘 et calculer l'inertie, ou score de distorsion, pour chaque modèle.
#  L'inertie mesure la similarité du cluster en calculant la distance totale entre les points de données et leur centre de cluster le plus proche. Les clusters avec des observations similaires ont tendance à avoir des distances plus petites entre eux et un score de distorsion plus faible dans l'ensemble.
# 
# ## La méthode du coude nous aidera à déterminer le nombre de groupes.
# 
# •Nous choisissons 'K' manuellement, par visualisation.
# 
# • Calculer les distances entre les points d'un cluster (With-in Cluster Sum of Squares "WCSS").
# 
# • Si nous minimisons 'WCSS', nous avons atteint la solution de clustering parfaite.
# 

# In[291]:


from sklearn.cluster import KMeans
from sklearn import cluster


K=range(1,10)
k_means = []
#On fait une boucle de 1 à 10 pour tester toutes ces possibiliéts
for k in K:
    #pour chaque k, on crée un modèle et on l’ajuste
    km=KMeans(n_clusters=k,init="k-means++").fit(X_scaled)
     #on stocke l’inertie associée
    k_means.append(km.inertia_)


#Visualisation des valeurs d'inertie pour chaque nombre de cluster
plt.plot(range(1, 10), k_means, marker='o')
plt.show()


# In[292]:


mycenters = pd.DataFrame({'groupe_km' : K, 'WSS' : k_means})
mycenters


# On remarque que le nombre de 5 Clusters n'est pas idéal pour le Kmeans. La meilleure alternative serait 2 Clusters. Si l'on veut partitionner un peu plus, il faudrait considérer un K = 3 ou 4.
# 
# il est conseillé de choisir k = 5 .
#  Un clustering  en 5 permettra de de comparer le partitionnement avec les groupes de la classification hiérarchique. Il est pertinent de comparer les deux méthodes sur le même nombre de clusters.
# 
# 

# In[293]:



#Clustering K-Means en 5 clusters
km = cluster.KMeans(n_clusters=5)
km.fit(X_scaled)
#Récupération des clusters attribués à chaque individu (classes d'appartenance)
clusters_km = km.labels_
clusters_km


# # Visualisation des clusters en ACP pour la projection des données
# 

# Le principe de **la réduction de dimension** est de réduire la complexité superflue d'un dataset en projetant ses données dans un espace de plus petite dimension .
# 
# Le principe  de **ACP** est de projeter nos données sur des axes appelés Composantes Principales, en cherchant à minimiser la distance entre nos points et leur projections. De cette manière on réduit la dimension préservant au maximum la variance de nos données. Pour **Préserver un maximum de variance pour optenir la projection qu'il soit la plus fidèle possible à nos données.**
#  
#  Analyse Pour trouver les axes de projection (xp): 
#  Pour faire ça dans point de vue mathématique on
#  1. On calcule la matrice de covariance des données 
#  2. On détermine les vecteurs propres de cette matrice : ce sont les Composantes Principales 
#  3. (On projette les données sur ces axes)
#  
# 
# 
#  
# **L'ACP (Analyse en Composante Principale) permettra une visualisation des clusters pays sur le premier plan factoriel (ou plus). Il deviendra alors facile de pouvoir appréhender le "comportement" des différents groupes.**

# PCA est un transformer ! 
# 1. Définir le nombre de composantes 
# 2. Transformer les données avec fit transform()
# 
# Il y a deux cas possibles pour choisir le nombre de composantes sur lesquels projeter nos données? et bien :
# 1. Celui dans lequel vous cherchez à visualiser vos données dans un espace de 2d ou 3D ,pour ça c'est très simple, le nombre de composants doit être égale à deux ou trois
# 2. Celui dans lequel vous cherchez à compresser vos données pour accélérer l'apprentissage de la machine sur des taches de classification ou de régression, pour ça il faut choisir le nombre de composantes de telle sorte à préserver entre 95 et 99 % de la variance de vos données.
# 
# 
# L’enjeu d’une ACP est de trouver le meilleur plan de projection ayant la plus grande inertie, c’est à dire limitant le plus la perte d’information originelle. Les 7 variables seront synthétisées en de nouvelles variables : PC1, PC2, etc...
# 
# Comme précédemment, une ACP ne peut se faire que si les données sont centrées et réduites (transformation pour que moyenne = 0, écart-type = 1).
# 
# ### Définir le nombre de composantes 

# In[294]:


import pca as pca
from sklearn import decomposition

#decomposition.PCA: Réduction de la dimensionnalité linéaire à l'aide de la décomposition en valeurs singulières des données pour les projeter dans un espace de dimension inférieure.
pca = decomposition.PCA().fit(X_scaled) #sklearn
X_projected = pca.transform(X_scaled)


#nous allons examiner quel est le pourcentage de variance préserver pour chacune de nos composantes.
scree = pca.explained_variance_ratio_*100      #Le paramètre pca.explained_variance_ratio_ renvoie un vecteur de la variance expliquée par chaque dimension.
#array([49.88630268, 24.51850599, 10.68333729,  5.72429344,  4.27437791,3.66079486,  1.25238783])

plt.bar(np.arange(len(scree))+1, scree)
plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')   
plt.xlabel("rang de l'axe d'inertie")
plt.ylabel("pourcentage d'inertie")
plt.title("Eboulis des valeurs propres")
plt.savefig("exports/Eboulis des valeurs propres.png")

plt.show()

#Pourcentage de variance expliquée par les composantes principales à l'aide de .explained_variance_ratio_
print(scree.cumsum())




# In[295]:


scree 


# Environ 75 % de la variance des données s'explique par ces deux premières composantes.
# La méthode du coude précise une forte représentation de nos variables sur les deux premières composantes principales, le premier axe factoriel.
# 

# # ACP - Cercle des corrélations

# In[296]:


def cerle_corr(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # initialisation de la figure
            #fig, ax = plt.subplots(figsize=(12,(n_comp*2)))
            #ax.set_aspect('equal', adjustable='box')
            fig=plt.figure(figsize=(12,12))
            fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9)
            ax=fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box')

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            else :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
        
            # affichage des flèches
            plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.savefig("exports/Cercle des corrélations (F1 et F2).png")

            plt.show(block=False)


# In[297]:


pcs = pca.components_
cerle_corr(pcs, 4, pca, [(0,1)], labels = np.array(df_clus.columns))


# Dans notre étude, le premier plan factoriel de l’ACP a permis de conserver 75 % de l’information totale.
# 
# Ici, l’axe PC2 reflète bien le taux de dépendance aux importations et le taux d’autosuffisance. Plus la valeur de l’axe 2 est positive, et plus le pays est importateur. Au contraire, plus les valeurs sont négatives, et plus le pays est autosuffisant avec sa production de viande de volailles et importe peu. 
# 
# De même, l'axe PC1 est une combinaison de la disponibilité totale de protéines et de calories, du PIB, de la proportion de protéines animales et de l'évolution de la population. 
# 
# Plus la forte croissance démographique d'un pays est importante, plus sa valeur positive sur l'axe PC1 est élevée. 
# 
# À l'inverse, plus la valeur de l'axe PC1 est négative, plus le PIB du pays est élevé, plus la consommation de protéines animales et la disponibilité alimentaire de protéines et de calories sont élevées. 
# 
# La protéine  est la variable qui contribue le plus à l'axe PC1. 
# Enfin, il existe une certaine relation entre les variables du PIB, la proportion de protéines animales et la disponibilité de protéines totales et de calories.  
# 
#  

# In[298]:


#pca = decomposition.PCA().fit(X_scaled) #sklearn
#X_projected = pca.transform(X_scaled)


#Coordonnées factorielles 
plt.figure(figsize=(20,20))
plt.subplot(122)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=km.labels_)
plt.xlabel('F{} ({}%)'.format(1, round(100*scree[0],1)))
plt.ylabel('F{} ({}%)'.format(2, round(100*scree[1],1)))
plt.title("Projection en 5 clusters des {} individus sur le 1er plan factoriel".format(X_projected.shape[0]))

plt.savefig("exports/projection_clusters.png")
plt.show()


# On peut d'ailleurs calculer les valeurs de ces variables synthétiques F1 et F2 qui pourraient remplacer les autres variables :
# 
# 

# In[299]:


X_projected


# In[300]:


#Calcul des composantes principales
#Ici seulement F1 et F2 seront utiles à l'interprétation attendue
X_projected = pca.transform(X_scaled)

df_facto = pd.DataFrame(X_projected, index=df_clus.index, columns=["F" + str(i+1) for i in range(7)]).iloc[:, :2]
df_facto.head() #Affichage des 5 premières lignes


# On obtient donc un tableau de 5 lignes et 7 colonnes, pourquoi ?
# Nous avons cinq groupes et sept variables.
# **Pour réduire les dimensions, nous avons besoin de pca**

# L'analyse sera plus fine en 5 clusters. De plus, la comparaison sera possible avec les 5 groupes identifiés lors du précédent partitionnement, le contexte nous oriente davantage vers un clustering en 5 partitions.
# 
# Maintenant, il est nécessaire de caractériser chacun de ces groupes selon nos 8 variables. La position des centroïdes de chacun des groupes indiquera le ou les meilleurs clusters. C'est l'avantage de procéder en K-Means, afin d'obtenir directement des valeurs centrées et réduites, facilitant l'analyse. ⬇️

# In[301]:


df_groupes_cah['groupe_km'] = clusters_km
gb = df_groupes_cah.groupby('groupe_km')
nk = gb.size()
print(nk)


# In[302]:


# Moyennes conditionnelles
mk = gb.mean()
mk


# In[303]:


# Ajout des variables synthétiques F1 et F2
df_boxkm = pd.merge(df_groupes_cah, df_facto, on="pays", how="left")
df_boxkm = df_boxkm.sort_values("groupe_km")
df_boxkm.head()


# In[414]:


#les centroïdes des groupes et leurs coordonnées dans chacune des dimensions.
les_centroïdes_groupes= df_boxkm.groupby('groupe_km').mean()
les_centroïdes_groupes.to_csv('exports/P5_centroïdes_groupe_km.csv', index=False)

les_centroïdes_groupes


# In[305]:


def boxplot_cluster_km(var):
    data_boxplot = []
    groupes_pays = df_boxkm["groupe_km"].unique()
    
    for groupe in groupes_pays :
        subset = df_boxkm[df_boxkm.groupe_km == groupe]
        data_boxplot.append(subset[var])

    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax1.boxplot(data_boxplot, notch=0, vert=1, whis=1.5, labels=["0", "1", "2", "3", "4"])

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.set_title(("Distribution de " + var +" par Cluster Kmeans"), fontsize=22)
    ax1.set_ylabel(var)
    ax1.set_xlabel("Cluster")
    ax1.set_xlim(0, len(data_boxplot) + 0.5)

    plt.show()


# In[306]:


boxplot_cluster_km('Évolution population (%)')


# In[307]:


boxplot_cluster_km('dispo_calories')


# In[308]:


boxplot_cluster_km('dispo_proteines')


# In[309]:


boxplot_cluster_km('ratio_prot_anim')


# In[310]:


boxplot_cluster_km('PIB_par_habitant')


# In[311]:


boxplot_cluster_km('taux_suffisance')


# In[312]:


boxplot_cluster_km('dep_import')


# In[313]:


boxplot_cluster_km('F1')


# In[314]:


boxplot_cluster_km('F2')


# # Sélections des pays sur groupes Kmeans.
# 
# Basé sur les mêmes critères qui ont été suivis lors de la sélection des pays par CAH (les groupes qui dépendent fortement des importations).
#   et le PIB par habitant est beaucoup plus élevé), 
# Les groupes de Kmeans à conserver sont ceux où F2 est supérieur à 1, soit les groupes 2, 3 et 4.
# Nous constatons que les groupes 3 et 4 atteignent un niveau élevé de consommation de protéines animales et ont un PNB élevé,

# In[315]:


select_clusters_kmeans = df_boxkm.groupby("groupe_km").mean().reset_index()
select_clusters_kmeans = df_boxkm[df_boxkm["F1"]<-1]["groupe_km"].unique()
select_clusters_kmeans


# In[329]:


select_clusters_kmeans = df_boxkm.groupby("groupe_km").mean().reset_index()
select_clusters_kmeans = df_boxkm[df_boxkm["F2"]>1]["groupe_km"].unique()
select_clusters_kmeans


# In[338]:



df_select_pays_kmeans = df_boxkm[df_boxkm["groupe_km"].isin([1,4]) == True]
df_select_pays_kmeans.to_csv('exports/Des groupe pays sur kmeans.csv', index=False)

df_select_pays_kmeans


# In[331]:


df_select_pays_kmeans.shape


# Ces groupes comptent 35 pays 
# 
#  Le Kmeans  nous a permis de choisir un meilleur pays. Ce sera notre groupe cible de deuxième niveau

# In[343]:


df_compare_cah_kmeans = pd.merge(df_select_pays_cah[['pays','groupe_cah']], df_select_pays_kmeans[['pays','groupe_km']],
                     on='pays', how='outer')
df_compare_cah_kmeans.to_csv('exports/Des groupe pays sur kmeans et CAH.csv', index=False)

df_compare_cah_kmeans


# # Liste des pays et recommandations

# In[340]:


df_select_pays_km_cah=df_compare_1.dropna()
df_select_pays_km_cah


# In[337]:


df_select_pays_km_cah.shape


# 
#   Dans le premier niveau, on va choisir les pays qui sont communs aux groupes du Kmeans  et CAH, on obtient 20 pays.
# 
# Liste des pays et recommandations Dans un premier temps, on suggère donc de cibler les pays de l'UE, pour leur proximité et la facilité des échanges commerciaux : l'Allemagne, le Danemark, la Suède ,Luxembourg et le Norvège. 
# 
# 
# Prudence avec le Royaume-Uni, puisqu'avec le Brexit, les échanges commerciaux avec l'UE sont actuellement compliqués. Pour autant, considérant les intérêts économiques mutuels, les récentes directives qui ont été prises pour favoriser ces échanges sont en notre faveur. 
# 
#  
# 
# De plus, il n'est pas préférable d'exporter vers l'Amérique car c'est l'un des pays les plus exportateurs de poulet au monde en plus du Brésil. 
# 
#  
# 
# Dans un second temps, on pourrait également cibler les pays comme Hong Kong, le Japon, Émirats arabes unis, ainsi que le Koweït. 
# 
#  
# 
# FAO - Poultry production - Marchés et commerce https://www.fao.org/poultry-production-products/aspects-socio-economiques/marches-et-commerce/fr/ 
# 
#  
# 
# "Le Brésil est le principal exportateur de viande de volaille, suivi par les États-Unis et les Pays-Bas. Les principaux pays importateurs sont la Chine, le Japon, le Mexique et le Royaume-Uni." 
# 
#  
# 
# "Les pays les moins avancés sont de plus en plus dépendants des importations de viande de volaille. Le niveau de leurs importations est passé de 3 pour cent en 1961 à environ 30 pour cent en 2013." 
# 
#  
# 
#  
# 
#  
# 
# https://www.fao.org/poultry-production-products/production/fr/ 
# 
#  
# 
# "Les États-Unis d'Amérique sont le plus grand producteur de viande de volaille à l’échelle de la planète: ils produisent en effet 17 pour cent de la production mondiale. Viennent ensuite la Chine et le Brésil." 
# 
#  
# 
# "Pour répondre à la demande croissante, la production de viande de volaille mondiale a bondi, passant de 9 à 132 millions de tonnes entre 1961 et 2019." 
# 
#  
# 
# "En 2019, la viande de volaille représentait environ 39 pour cent de la production mondiale de viande." 
# 
#  
# 
# "Dans les pays en développement, environ 80 pour cent des ménages ruraux élèvent des volailles." 

# # Tests statistiques
# 
# 
# Dans notre partition, nous avons des groupes séparés. Pour vérifier qu'ils diffèrent réellement. Nous avons effectué les test statistique  Kolmogorov-Smirnov qui est un test d'adéquation : parmi les sept variables, nous recherchons une variable avec une distribution normale.
# On peut tester l’adéquation de la 'Disponibilité alimentaire de prot (g/personne/jour)' à une loi normale .
# 
# 
# 
# Pour évaluer si cet échantillon peut être considéré comme gaussien, on peut étudier l'écart entre la fonction de répartition d'une loi normale et celle estimée de notre échantillon : la fonction de répartition empirique !
# Plus cette quantité est grande, plus on est enclin à rejeter l'hypothèse comme quoi l'échantillon est gaussien.
# 

# In[158]:


import scipy.stats as st
from scipy import stats
from scipy.stats import ks_2samp


# ### Test d'adéquation de Kolmogorov-Smirnov :

# ### Vérification des hypothèses

# **H0 = La variable suit donc  une loi normale .**
# 
# **H1 = La variable ne suit pas une loi normale.**

# In[159]:


df_groupes_cah.head(1)


# In[160]:


#Kolmogorov Smirnov test
stat, p= st.ks_2samp(df_groupes_cah['dispo_proteines'], 
            np.random.normal(df_groupes_cah['dispo_proteines'].mean(), 
                             df_groupes_cah['dispo_proteines'].std(ddof=0),
                             df_groupes_cah['dispo_proteines'].count()))

print('Statistics=%.3f, p=%.3f' % (stat, p))

#Interprétation
alpha = 0.05
if p > alpha:
    print(' Nous pouvons accepter H0 pour des niveaux de test de 5 %')
else:
    print('H0 est rejetée à un niveau de test de 5%')
    


# ### Test d'adéquation de Shapiro-Wilk :
# Le test sera doublé par celui de Shapiro-Wilk.
# Nous avons échantillonné les deux groupes de pays sélectionnés aux CAH 4 et 5
# Comme nous avions un petit échantillon on peux faire Le test  Shapiro-Wilk.
# Le test de Shapiro-Wilk est un test statistique utilisé pour vérifier si une variable continue suit une distribution normale. L'hypothèse nulle (H0) indique que la variable est normalement distribuée, et l'hypothèse alternative (H1) indique que la variable n'est PAS normalement distribuée. Donc après avoir exécuté ce test :
# 
# Si p ≤ 0,05 : alors l'hypothèse nulle peut être rejetée (c'est-à-dire que la variable n'est PAS distribuée normalement).
# Si p > 0,05 : alors l'hypothèse nulle ne peut pas être rejetée (c'est-à-dire que la variable PEUT ÊTRE normalement distribuée).
# 

# In[161]:


#creation of the df with only clusters 4 & 5
c4c5 = df_groupes_cah[(df_groupes_cah['groupe_cah'] == 4) | (df_groupes_cah['groupe_cah'] == 5)]


# In[163]:


#normality of variables in c4c5
import pingouin as pg
pg.normality(c4c5, method='shapiro', alpha=0.05).drop('groupe_cah')

#normality: test de normalité univarié.


# Nous constatons que les variables sont dispo_calories, dispo_proteines, ratio_proteines_animales(%), et PIB_par habitant. distribué selon la loi normale.

# In[164]:


#histogram
sns.histplot(data=c4c5, x='dispo_proteines', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité dispo_proteines.jpg")

plt.show()


# #### Disponibilité alimentaire énergétique
# 

# In[165]:


#histogram
sns.histplot(data=c4c5, x='dispo_calories', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité dispo_calories.jpg")
plt.show()


# #### Importation de viande de volaille
# 

# In[166]:


#histogram
sns.histplot(data=c4c5, x='Évolution population (%)', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité Évolution populations.jpg")

plt.show()


# #### Pourcentage de protéine animale
# 

# In[167]:


#histogram
sns.histplot(data=c4c5, x='ratio_prot_anim', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité ratio_pro.jpg")

plt.show()


# #### Produit Intérieur Brut
# 

# In[168]:


#histogram
sns.histplot(data=c4c5, x='PIB_par_habitant', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité PIB.jpg")

plt.show()


# #### Importation de viande de volaille
# 

# In[169]:


#histogram
sns.histplot(data=c4c5, x='taux_suffisance', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité taux_suffisance.jpg")

plt.show()


# In[170]:


#histogram
sns.histplot(data=c4c5, x='dep_import', kde=True, color='#4cb2ff')
plt.savefig("exports/tester la normalité taux_dépendance_importations.jpg")

plt.show()


# ### Test de comparaison de deux clusters dans le cas gaussien.
# 
# Si on souhaite comparer deux échantillons (i.i.d) gaussiens, il nous suffit en fait de comparer leurs paramètres : leur moyenne μ1 et μ2, et leur variance σ21 et σ22.
# La méthodologie la plus classique est d'effectuer de manière séquentielle :
# 
# Un test d'égalité des variances.
# 
# Un test d'égalité des moyennes.
# 
# Si les variances ne sont pas considérées comme égales, les deux échantillons n'ont pas la même loi. Si les variances sont considérées comme égales, il est alors possible d'estimer cette variance sur les deux échantillons à la fois, et de tester l'égalité des moyennes en utilisant cette variance empirique globale.
# Notons qu'il est néanmoins possible d'effectuer un test de comparaison des moyennes sous hypothèse de variances différentes. Il ne s'agit pas d'une comparaison des lois, mais alors d'une comparaison simple des moyennes.
# 
# 

# ## La variable 'dispo_proteines' suit une loi normale et sera par conséquent choisie pour le test.
# 
# 

# In[171]:


#boxplot
color = sns.color_palette('pastel')

plt.figure(figsize=(30,15)) 

sns.boxplot(data=c4c5[(c4c5["groupe_cah"]== 4) | (c4c5["groupe_cah"]== 5)], x='dispo_proteines', y='groupe_cah', orient='h', palette=color,
            fliersize=4 , showfliers=True, showmeans=True, meanprops={"marker":"o", 
                                                                      "markerfacecolor":"white",
                                                                      "markeredgecolor":"white", 
                                                                      "markersize":"10"})

plt.xlabel('Dispo alim en protéine', fontsize=20, labelpad=30, fontweight='bold')
plt.xticks(fontsize=20)
plt.ylabel('Cluster', fontsize=20, labelpad=30, fontweight='bold')
plt.yticks(fontsize=20)
plt.title('Disponibilité alimentaire en protéine pour les cluster 4 et 5', fontsize=35, pad=50)
plt.savefig("exports/Disponibilité alimentaire en protéine pour les cluster 4 et 5.jpg")


plt.show()


# ### Tester l'égalité de la variance
# 
# **H0 = Les variance sont égales .**
# 
# **H1 = Les variance ne sont pas égales.** 

# In[172]:


#On teste tout d’abord l’égalité des variances à l’aide de la commande
pg.homoscedasticity(df_groupes_cah, dv='dispo_proteines',  group='groupe_cah', method='levene', 
                    alpha=0.05)
#Thomoscedasticity :tester l'égalité de la variance.


# ### Tester l'égalité des moyennes
# 
# **H0 = Les moyennes sont égales .**
# 
# **H1 = Les moyennes ne sont pas égales.** 

# In[173]:


#On teste ensuite l’égalité des moyennes à l’aide de la commande
pg.ttest(df_groupes_cah['dispo_proteines'][df_groupes_cah['groupe_cah'] == 4],
         df_groupes_cah['dispo_proteines'][df_groupes_cah['groupe_cah'] == 5],
         paired=False,
        
         confidence=0.95)


# In[174]:


α = 0.05

if α > pg.ttest(df_groupes_cah['dispo_proteines'][df_groupes_cah['groupe_cah'] == 4],
             df_groupes_cah['dispo_proteines'][df_groupes_cah['groupe_cah'] == 5]).iloc[0,3] : 
    
    print("La p-value étant inférieure au risque α, on rejette donc H0, les moyennes des deux groupes sont différentes.")
else :
    print("La p-value étant supérieur au risque α, H0 est donc vrai, les moyennes des deux groupes sont égales.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




