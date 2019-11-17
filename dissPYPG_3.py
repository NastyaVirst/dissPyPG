import pandas as pd
import psycopg2
import json
import vi_utils
import dsp
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score
#устанавливаем соединение с БД
#conn = psycopg2.connect(dbname='n_test', user='nastya', 
#                       password='12345', host='localhost')
print("begin")
engine = create_engine('postgresql+psycopg2://nastya:12345@localhost/n_test')
conn = engine.connect()
print("stage # 4")
#Подготовка кластеризации - определение числа кластеров
df_cluster_tmp = pd.read_sql('''select id,p_nas,t_plast, por, press_p, density,viscosity,
                        press_oil,paraf, brimstone, pronis, koeff_satur, oil_square,
                        thick_eff, thick_main, sm_asfal, koeff_sancliness, number_inter_well,
                        dist_water_well, aver_inter_r, fact_acceleration, buf_press, v_zak_water,
                        v_oil, v_water, v_liquid, koeff_ohv, t_work, zab_press,
                        water_cut, koeff_prod from virsta.welldata_norm''',conn)

split_data = dsp.DataSplitInfo(df_cluster_tmp)
df_cluster = split_data.columns_data

km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter = 300,
            tol=1e-04)
y_km = km.fit_predict (df_cluster)
cluster_labels = np.unique(y_km) #номера кластеров, в данном случае 0, 1 и 2, так как всего их 3
print ('y_km')

print("clear cluster_num")
conn.execute("delete from virsta.welldata_cluster")
print("insert cluster_num")
ln = len(y_km)
for i in range(ln):
    print("cluster_num {0}/{1}".format(i,ln) )
    conn.execute("INSERT INTO virsta.welldata_cluster (data_id, cluster_num) VALUES ({0}, {1})".format(split_data.ids[i],y_km[i]))

n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(df_cluster,
                                    y_km,
                                    metric='euclidean') +0.5
y_ax_lower =0
y_ax_upper = 0
yticks = []
for i, с in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == с]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters) 
    plt.barh(range (y_ax_lower , y_ax_upper),
             c_silhouette_vals ,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
#среднее
plt.axvline(silhouette_avg,
            color="red",
            linestyle= "--")
plt.yticks(yticks , cluster_labels + 1)
plt.ylabel('Кластер')
plt.xlabel('Силуэтный коэффициент ' )
plt.show()

distortions =[]
distortions_sil =[]
max_cluster = 8
for i in range(2, max_cluster):
    print('Cluster - {0}'.format(i))
    km = KMeans(n_clusters=i, init = 'k-means++',n_init=10,max_iter=300,n_jobs=-1)
    #для метода локтя
    km.fit(df_cluster)
    distortions.append(km.inertia_)
    #для силуэтного метода
    cluster_labels = km.fit_predict(df_cluster) 
    silhouette_avg2 = silhouette_score(df_cluster, cluster_labels) 
    distortions_sil.append(silhouette_avg2+0.5)
   
plt.plot(range(2,max_cluster), distortions, marker='o')
plt.xlabel('Чиcлo кластеров')
plt.ylabel('Искажение ' )
plt.show()

plt.plot(range(2,max_cluster), distortions_sil, marker='o')
plt.xlabel('Чиcлo кластеров')
plt.ylabel('Среднее значение силуэтного коэффициента')
plt.show()

print("done")

#print(len(d.keys()))


        
