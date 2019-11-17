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

print("stage # 3")
#Этап 3. Преобразование Бокса-Кокса и нормализация данных
#Пока что данные берем все, потом уберем выбросы (закомментировано в запросе)
df_box_norm = pd.read_sql('''select id, p_nas,t_plast, por, press_p, density,viscosity,
                        press_oil,paraf, brimstone, pronis, koeff_satur, oil_square,
                        thick_eff, thick_main, sm_asfal, koeff_sancliness, number_inter_well,
                        dist_water_well, aver_inter_r, fact_acceleration, buf_press, v_zak_water,
                        v_oil, v_water, v_liquid, koeff_ohv, t_work, zab_press,
                        water_cut, koeff_prod from virsta.welldata --where id not in (select id from well_iso_tree where tree_status = -1)''',conn)

columns_data = {}
columns_data_ori = {}

print("stage # 3.1")

for col in df_box_norm:  
    temp = [] 
    sc = StandardScaler()
    for i in df_box_norm[col].iteritems():
        temp.append(i[1])
    print (col)
    columns_data_ori[col] = temp
    if col != 'id':
        transfbox, lmbda = boxcox(temp) 
        transfbox = transfbox.reshape(-1,1)
        sc.fit(transfbox)
        data_box_norm=sc.transform(transfbox)
        columns_data[col] = data_box_norm.tolist()
    else:
        columns_data[col] = temp
       # print(data_box_norm)

print("generate inserts")
insrts = vi_utils.ToInsertLines(columns_data)
#with open('inserts.sql', 'w') as f:
#    for item in insrts:
#        f.write("%s;\n" % item)

print("clear")
conn.execute("delete from virsta.welldata_norm")
print("insert")
ln = len(insrts)
for i in range(ln):
    print("norm {0}/{1}".format(i,ln) )
    conn.execute(insrts[i])

print("done")

#print(len(d.keys()))