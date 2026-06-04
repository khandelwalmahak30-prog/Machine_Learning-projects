                                              ##principal component analysis(PCA)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data={
    'age':[25,30,35,40,50,60],
    'income':[30000,40000,50000,60000,70000,8000],
    'spending':[70,60,50,40,30,20],
    'saving':[1000,5000,8000,1000,15000,20000]
}
df=pd.DataFrame(data)
scaler=StandardScaler()
scaler_data=scaler.fit_transform(df)
pca=PCA(n_components=2)
pca_result=pca.fit_transform(scaler_data)
pca_df=pd.DataFrame(pca_result,columns=['PCA1','PCA2'])

explained_variance=pca.explained_variance_ratio_
print("variance captured by each pca comoponent:")
print(np.round(explained_variance*100,2))

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'],pca_df['PCA2'],color='black',s=80) # Corrected: plot PCA1 against PCA2
plt.title("PCA projection (2d view)")
plt.xlabel('PCA1 main pattern')
plt.ylabel('PCA2 main pattern')
plt.grid(True)
plt.show()
print("new data with new features PCA1,PCA2")
print(pca_df)
print(df)