import os
import random
from datetime import datetime, date, timedelta
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import tqdm
from generate_transformers import HoroModel


def generate_for_dates(start_date = date(2021, 1, 1), end_date = date(2021, 1, 5), template = "planets_template.csv"):
  COLUMNS = ["овен",
      "телец",
      "близнецы",
      "рак",
      "лев",
      "дева",
      "весы",
      "скорпион",
      "козерог",
      "стрелец",
      "водолей",
      "рыбы",
      "main_horo",
      "date"
  ]
  SIGNS_DICT = {
    '0':"main_horo",
    '1':"овен",
    '2':"телец",
    '3':"близнецы",
    '4':"рак",
    '5':"лев",
    '6':"дева",
    '7':"весы",
    '8':"скорпион",
    '9':"козерог",
    '10':"стрелец",
    '11':"водолей",
    '12':"рыбы",
  }

  SIGNS = ["овен",
      "телец",
      "близнецы",
      "рак",
      "лев",
      "дева",
      "весы",
      "скорпион",
      "козерог",
      "стрелец",
      "водолей",
      "рыбы",
  ]
  PLANETS = {'Sun': 10,  'Mercury': 199, 'Venus': 299,
              'Mars': 499, 'Jupiter': 599, 'Saturn': 699,
              'Uranus': 799, 'Neptune': 899, 'Pluto': 999}

  def extract_features_packet(date, template):
      date_for_epoch = Time(date).jd
      df_planets = pd.read_csv(template, sep=";")
      features_df = pd.DataFrame()
      if "Unnamed: 0" in df_planets.columns:
        del df_planets["Unnamed: 0"]
      for k, v in PLANETS.items():
          kk = [k+'_x',k+'_y',k+'_z',k+'_vx',k+'_vy',k+'_vz',k+'_l',k+'_ry',k+'_rr']
          df_planets.loc[0, kk] = Horizons(id=v, location=500, epochs=date_for_epoch, id_type='id').vectors().to_pandas()[['x','y','z','vx','vy','vz','lighttime','range','range_rate']].values[0]
      for sign in SIGNS:
        cls_cols = [el.capitalize() + '_cls' for el in SIGNS]
        df_ohe = pd.DataFrame(columns=cls_cols)
        df_ohe.loc[0] = [0,0,0,0,0,0,0,0,0,0,0,0]
        df_ohe[sign.capitalize() + '_cls'] = 1
        feature_df = pd.concat([df_planets,df_ohe],axis=1)
        features_df = features_df.append(feature_df)
      return(features_df)

  time_between_dates = end_date - start_date
  days_between_dates = time_between_dates.days

  for i in tqdm.notebook.tqdm(range(days_between_dates)):
    cur_date = str(start_date + timedelta(days=i))
    if i == 0:
      res_df = extract_features_packet(cur_date,template)
    else:
      res_df = pd.concat([res_df,extract_features_packet(cur_date,template)],axis=0)


  test_model = HoroModel(gen_path='/content/gdrive/My Drive/MADE_graduation_project_ML_horoscopes/GANs_results/GAN300_Lin_Lin_SentBySent_eval_norm/gen_500000',
                        #gpt_path ='/content/gdrive/MyDrive/medium_cleaned',
                        gpt_path ='/content/gdrive/MyDrive/small',
                        scaler_path = '/content/datatransformer.pickle',
                        kv_path = "/content/gdrive/My Drive/MADE_graduation_project_ML_horoscopes/data/model_rusvectores.model")

  res_avg = test_model.get_prediction(res_df.to_numpy(), mode = 'with_avg')
  horoscopes = []
  for i in tqdm.notebook.tqdm(range(days_between_dates)):
    cur_date = str(start_date + timedelta(days=i))
    row = res_avg[i*12:(i+1)*12]
    row.append(res_avg[len(res_avg)-days_between_dates+i])
    row.append(cur_date)
    horoscopes.append(row)
  horo_df = pd.DataFrame(horoscopes, columns=['овен','телец','близнецы','рак','лев','дева','весы','скорпион','стрелец','козерог','водолей','рыбы','main_horo','date'])
  horo_df.to_csv(str(end_date) + '_results.csv',sep = ';', index = False)

  test_model_otvety = HoroModel(gen_path='/content/gdrive/My Drive/MADE_graduation_project_ML_horoscopes/GANs_results/GAN300_Lin_Lin_SentBySent_eval_norm/gen_500000',
                        #gpt_path ='/content/gdrive/MyDrive/medium_cleaned',
                        gpt_path ='/content/gdrive/MyDrive/medium_cleaned',
                        scaler_path = '/content/datatransformer.pickle',
                        kv_path = "/content/gdrive/My Drive/MADE_graduation_project_ML_horoscopes/data/model_rusvectores.model")

  res_otvety_avg = test_model_otvety.get_prediction(res_df.to_numpy(), mode = 'with_avg')
  otvety_horoscopes = []
  for i in tqdm.notebook.tqdm(range(days_between_dates)):
    cur_date = str(start_date + timedelta(days=i))
    row = res_otvety_avg[i*12:(i+1)*12]
    row.append(res_otvety_avg[len(res_otvety_avg)-days_between_dates+i])
    row.append(cur_date)
    otvety_horoscopes.append(row)
  otvety_df = pd.DataFrame(otvety_horoscopes, columns=['овен','телец','близнецы','рак','лев','дева','весы','скорпион','стрелец','козерог','водолей','рыбы','main_horo','date'])
  otvety_df.to_csv(str(end_date) + '_otvety_results.csv', sep = ';', index = False)