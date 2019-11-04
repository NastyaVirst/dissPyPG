import pandas as pd
import psycopg2
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import sparse as sc
#устанавливаем соединение с БД
#conn = psycopg2.connect(dbname='n_test', user='nastya', 
#                       password='12345', host='localhost')
print("begin")
engine = create_engine('postgresql+psycopg2://nastya:12345@localhost/n_test')
conn = engine.connect()

print("stage # 1")
#Этап 1. Заполнение пропусков средним значением
#из данных БД делаем табличный тип DataFrame df
#оригинальные данные
df_orig = pd.read_sql ('''SELECT 
                        w.id, w.p_nas,w.t_plast, w.por, w.press_p, w.density,w.viscosity,
                        w.press_oil,w.paraf, w.brimstone, w.pronis, w.koeff_satur, w.oil_square,
                        w.thick_eff, w.thick_main, w.sm_asfal, w.koeff_sancliness, w.number_inter_well,
                        w.dist_water_well, w.aver_inter_r, w.fact_acceleration, w.buf_press, w.v_zak_water,
                        w.v_oil, w.v_water, w.v_liquid, w.koeff_ohv, w.t_work, w.zab_press,
                        w.water_cut, w.koeff_prod
                        FROM virsta.v_welldata w
                        where w is not null''',conn) 
print("stage # 2")
#изменяемые данные
df = pd.read_sql('''SELECT 
                    w.id, w.p_nas,w.t_plast, w.por, w.press_p, w.density,w.viscosity,
                    w.press_oil,w.paraf, w.brimstone, w.pronis, w.koeff_satur, w.oil_square,
                    w.thick_eff, w.thick_main, w.sm_asfal, w.koeff_sancliness, w.number_inter_well,
                    w.dist_water_well, w.aver_inter_r, w.fact_acceleration, w.buf_press, w.v_zak_water,
                    w.v_oil, w.v_water, w.v_liquid, w.koeff_ohv, w.t_work, w.zab_press,
                    w.water_cut, w.koeff_prod
                    FROM virsta.welldata w''',conn) 
#обрабатываем пропущенные значения заменой средним-mean
print("stage # 3")
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer = imputer.fit(df_orig)
imputed_data = imputer.transform(df)
df = pd.DataFrame(imputed_data)
#записываем данные во временную таблицу
df.to_sql('welldata_tmp', conn, schema = 'virsta', if_exists='replace',index=False)
#переписываем данные из временной в основную
rs = pd.read_sql('''with t as(update welldata set (id, p_nas,t_plast, por, press_p, density,viscosity,
                        press_oil,paraf, brimstone, pronis, koeff_satur, oil_square,
                        thick_eff, thick_main, sm_asfal, koeff_sancliness, number_inter_well,
                        dist_water_well, aver_inter_r, fact_acceleration, buf_press, v_zak_water,
                        v_oil, v_water, v_liquid, koeff_ohv, t_work, zab_press,
                        water_cut, koeff_prod) = (select * from welldata_tmp 
                        where welldata_tmp."0" = welldata.id) returning  id, p_nas,t_plast,
                        por, press_p, density,viscosity,
                        press_oil,paraf, brimstone, pronis, koeff_satur, oil_square,
                        thick_eff, thick_main, sm_asfal, koeff_sancliness, number_inter_well,
                        dist_water_well, aver_inter_r, fact_acceleration, buf_press, v_zak_water,
                        v_oil, v_water, v_liquid, koeff_ohv, t_work, zab_press,
                        water_cut, koeff_prod) 
                    select * from t order by t.id asc''',conn)
#выводим количество обработанных строк
#print(rs.rowcount)
print("stage # Z")
#Этап №2. Определение выбросов методом Isolation Forest
#Функция для преобразования dataframe в разреженную матрицу csr-matrix
#d = pd.DataFrame(rs)
#создадим справочник соответствий id записи в бд и номер строки
kk=[]
vv=[]
for i, row in rs.iterrows():
    ss=[]
    for j, column in row.iteritems():
        if j!='id':
            ss.append(column)
        else:
            kk.append(column)
    vv.append(ss)
#print(vv )

print("begin tree")
clf = IsolationForest(behaviour='new', max_samples=3000, contamination='auto')
clf.fit(vv)
y_pred = clf.predict(vv)

print("clear")
conn.execute("delete from well_iso_tree")
print("insert")
ln = len(y_pred)
for i in range(ln):
    print("{0}/{1}".format(i,ln) )
    conn.execute("INSERT INTO virsta.well_iso_tree (id, tree_status) VALUES  ({0}, {1})".format(kk[i],y_pred[i] ))
#Этап 3. Преобразование Бокса-Кокса и нормализация данных
#Пока что данные берем все, потом уберем выбросы (закомментировано в запросе)
df_box_norm = pd.read_sql('''select id, p_nas,t_plast, por, press_p, density,viscosity,
                        press_oil,paraf, brimstone, pronis, koeff_satur, oil_square,
                        thick_eff, thick_main, sm_asfal, koeff_sancliness, number_inter_well,
                        dist_water_well, aver_inter_r, fact_acceleration, buf_press, v_zak_water,
                        v_oil, v_water, v_liquid, koeff_ohv, t_work, zab_press,
                        water_cut, koeff_prod from virsta.welldata --where id not in (select id from well_iso_tree where tree_status = -1)''',conn)
for col in df_box_norm:  
    temp = [] 
    sc = StandardScaler()
    for i in df_box_norm[col].iteritems():
        if col != 'id':
            temp.append(i[1])
    print (col)
    if len(temp)!=0:
        transfbox, lmbda = boxcox(temp) 
        transfbox = transfbox.reshape(-1,1)
        sc.fit(transfbox)
        data_box_norm=sc.transform(transfbox)
        print(data_box_norm)
print("done")

#print(len(d.keys()))


        
