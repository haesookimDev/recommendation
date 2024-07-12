import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA

class PreprocessingRawData():
    def preprocessing(self, TMA, TA, VAI):
        PRE_TA = pd.DataFrame()
        PRE_TMA = pd.DataFrame()
        PRE_VAI = pd.DataFrame()
        print("Preprocessing Raw Data")
        columns=['SHOPPING',
                'PARK',
                'HISTORY',
                'TOUR',
                'SPORTS',
                'ARTS',
                'PLAY',
                'CAMPING',
                'FESTIVAL',
                'SPA',
                'EDUCATION',
                'DRAMA',
                'PILGRIMAGE',
                'WELL',
                'SNS',
                'HOTEL',
                'NEWPLACE',
                'WITHPET',
                'MIMIC',
                'ECO',
                'HIKING']
        
        print("Copying Raw Data")
        PRE_TMA: pd.DataFrame = TMA[['TRAVELER_ID',
                    'GENDER',
                    'AGE_GRP',
                    'TRAVEL_LIKE_SIDO_1',
                    'TRAVEL_LIKE_SGG_1',
                    'TRAVEL_LIKE_SIDO_2',
                    'TRAVEL_LIKE_SGG_2',
                    'TRAVEL_LIKE_SIDO_3',
                    'TRAVEL_LIKE_SGG_3',
                    'TRAVEL_STYL_1',
                    'TRAVEL_STYL_2',
                    'TRAVEL_STYL_3',
                    'TRAVEL_STYL_4',
                    'TRAVEL_STYL_5',
                    'TRAVEL_STYL_6',
                    'TRAVEL_STYL_7',
                    'TRAVEL_STYL_8',
                    'TRAVEL_STATUS_DESTINATION',
                    'TRAVEL_MOTIVE_1',
                    'TRAVEL_MOTIVE_2',
                    'TRAVEL_MOTIVE_3']].copy()

        PRE_TA: pd.DataFrame = TA[['TRAVELER_ID', 'TRAVEL_ID', 'TRAVEL_PURPOSE']].copy()

        PRE_VAI = VAI[['VISIT_AREA_ID', 'TRAVEL_ID', 'VISIT_ORDER', 'VISIT_AREA_NM',
        'VISIT_START_YMD', 'VISIT_END_YMD', 'ROAD_NM_ADDR', 'LOTNO_ADDR',
        'X_COORD', 'Y_COORD', 'RESIDENCE_TIME_MIN', 'VISIT_AREA_TYPE_CD', 'REVISIT_YN',
        'VISIT_CHC_REASON_CD', 'DGSTFN', 'REVISIT_INTENTION',
        'RCMDTN_INTENTION']].copy()

        print("Encoding Traveler Data")

        
        PRE_TMA['GENDER'] = PRE_TMA['GENDER'].apply(encoding_gender)
        PRE_TMA['TRAVEL_STATUS_DESTINATION'] = PRE_TMA['TRAVEL_STATUS_DESTINATION'].apply(encoding_destination)
        

        print("Encoding Trip Data")
        PRE_TA['TRAVEL_PERIOD'] = TMA['TRAVEL_STATUS_YMD'].apply(tripdaycheck)
        PRE_TA = pd.concat([PRE_TA, pd.DataFrame(data=TA['TRAVEL_START_YMD'].apply(split_YMD).tolist(), columns=['Y','M','D'])], axis=1)
        PRE_TA = pd.concat([PRE_TA, pd.DataFrame(data=PRE_TA['TRAVEL_PURPOSE'].apply(lambda x:encoding_purpose(x)).tolist(), columns=columns)], axis=1)
        

        print("Encoding Dest Data")
        
        PRE_VAI = pd.concat([PRE_VAI, pd.DataFrame(data=VAI['VISIT_START_YMD'].apply(split_YMD).tolist(), columns=['Y','M','D'])], axis=1)
        PRE_VAI = pd.concat([PRE_VAI, pd.DataFrame(data=VAI.fillna(" ").apply(lambda x: split_sg(x), axis=1).tolist(), columns=['S','G'])], axis=1)
        PRE_VAI['REVISIT_YN'] = PRE_VAI['REVISIT_YN'].apply(encoding_revisit)
        PRE_VAI=PRE_VAI.drop(PRE_VAI[
            (PRE_VAI['VISIT_AREA_TYPE_CD']==21)|
            (PRE_VAI['VISIT_AREA_TYPE_CD']==22)|
            (PRE_VAI['VISIT_AREA_TYPE_CD']==23)|
            (PRE_VAI['VISIT_AREA_TYPE_CD']==24)
            ].index
                            )
        
        PRE_VAI=PRE_VAI.drop(PRE_VAI[
            (PRE_VAI['S']==0)|
            (PRE_VAI['G']==0)
            ].index)
        
        print("Prepare Data before Using for Model")
        PRE_TMA.fillna(0, inplace=True)
        PRE_TMA['TRAVEL_STYL'] = reduce_feature(PRE_TMA[['TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']])
        PRE_TMA['TRAVEL_MOTIVE'] = reduce_feature(PRE_TMA[['TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3']])

        return PRE_TMA.fillna(0), PRE_TA.fillna(0), PRE_VAI.fillna(0)

PATH_DATA="../Preprocessing/code/"

codeA = pd.read_csv(PATH_DATA+"tc_codea_코드A.csv")
codeB = pd.read_csv(PATH_DATA+"tc_codeb_코드B.csv")
sgg = pd.read_csv(PATH_DATA+"tc_sgg_시군구코드.csv")

cd=sgg['SGG_CD1'].unique().tolist()
name = sgg['SIDO_NM'].unique().tolist()

for idx, i in enumerate(name):
  nm = i[:2]
  if nm in ['충청','전라','경상']:
    if i =='충청북도':
      name[idx]='충북'
    elif i =='충청남도':
      name[idx]='충남'
    elif i =='전라북도':
      name[idx]='전북'
    elif i =='전라남도':
      name[idx]='전남'
    elif i =='경상북도':
      name[idx]='경북'
    elif i =='경상남도':
      name[idx]='경남'
  else:
    name[idx]=nm

dest_map = dict(zip(name, cd))

dest_map['도서 지역']=51

TP_CD_DICT = {0 : 'None',
              1 : 'SHOPPING',
              2 : 'PARK',
              3 : 'HISTORY',
              4 : 'TOUR',
              5 : 'SPORTS',
              6 : 'ARTS',
              7 : 'PLAY',
              8 : 'CAMPING',
              9 : 'FESTIVAL',
              10 : 'SPA',
              11 : 'EDUCATION',
              12 : 'DRAMA',
              13 : 'PILGRIMAGE',
              21 : 'WELL',
              22 : 'SNS',
              23 : 'HOTEL',
              24 : 'NEWPLACE',
              25 : 'WITHPET',
              26 : 'MIMIC',
              27 : 'ECO',
              28 : 'HIKING'}    



def encoding_purpose(x:str):
  
  TP_DICT = {'SHOPPING': 0,
              'PARK': 0,
              'HISTORY': 0,
              'TOUR': 0,
              'SPORTS': 0,
              'ARTS': 0,
              'PLAY': 0,
              'CAMPING': 0,
              'FESTIVAL': 0,
              'SPA': 0,
              'EDUCATION': 0,
              'DRAMA': 0,
              'PILGRIMAGE': 0,
              'WELL': 0,
              'SNS': 0,
              'HOTEL': 0,
              'NEWPLACE': 0,
              'WITHPET': 0,
              'MIMIC': 0,
              'ECO': 0,
              'HIKING': 0}

  TP_STRING = x.split(';')

  for i in TP_STRING:
      if i.isdigit():
          TP_DICT[TP_CD_DICT[int(i)]]=1
      else:
          TP_DICT['None']=1
  return [TP_DICT['SHOPPING'],
          TP_DICT['PARK'],
          TP_DICT['HISTORY'],
          TP_DICT['TOUR'],
          TP_DICT['SPORTS'],
          TP_DICT['ARTS'],
          TP_DICT['PLAY'],
          TP_DICT['CAMPING'],
          TP_DICT['FESTIVAL'],
          TP_DICT['SPA'],
          TP_DICT['EDUCATION'],
          TP_DICT['DRAMA'],
          TP_DICT['PILGRIMAGE'],
          TP_DICT['WELL'],
          TP_DICT['SNS'],
          TP_DICT['HOTEL'],
          TP_DICT['NEWPLACE'],
          TP_DICT['WITHPET'],
          TP_DICT['MIMIC'],
          TP_DICT['ECO'],
          TP_DICT['HIKING']]

def tripdaycheck(x: str)->int:
    x_1 = x.split('~')
    start = x_1[0].split('-')
    end = x_1[1].split('-')
    start = datetime(int(start[0]), int(start[1]), int(start[2]))
    end = datetime(int(end[0]), int(end[1]), int(end[2]))
    diff = end-start

    return diff.days+1

#날짜를 연/월/일로 분리
def split_YMD(x: str)->int:
    YMD = x.split('-')

    return [int(YMD[0]), int(YMD[1]), int(YMD[2])]

#주소에서 시/도, 군/구만 추출
def split_sg(x)->str:
    x_1 =x['LOTNO_ADDR']
    x_2 =x['ROAD_NM_ADDR']
    if x_1 !=" ":
      if x_1 !=None:
          sigg = x_1.split(" ")
      else:
          sigg=[""]
    elif x_2 !=" ":
      if x_2 !=None:
          sigg = x_2.split(" ")
      else:
          sigg=[""]
    else:
      sigg=[""]
    #시도, 군/구 제거 ex> 강남구 > 강남
    if sigg[0] in dest_map.keys():
      if len(sigg)>=2:
        if ('도' in sigg[0])|('시' in sigg[0] ):
          sigg[0] = sigg[0][:-1]
        if len(sigg)>=3:
          if ('구' in sigg[2][-1]):
            sigg[1] = ' '.join(sigg[1:3])
        try:
          return [int(dest_map[sigg[0]]), int(sgg.loc[(sgg['SGG_NM']==sigg[1])&(sgg['SGG_CD1']==dest_map[sigg[0]])]['SGG_CD2'].unique()[0])]
        except:
          return [0, 0]
      else:
        return [0, 0]
    else:
      return [0, 0]
    
def encoding_revisit(x: str)->int:
  if x=='Y':
    return 2
  elif x=='N':
    return 1
  else:
    return 0

def encoding_gender(x: str)->int:
  if x=='남':
    return 1
  elif x=='여':
    return 2
  else:
    return 3

def encoding_destination(x: str)->int:
  return dest_map[x]

def reduce_feature(data):
    pca = PCA(n_components=1)
    reduced_data = pca.fit_transform(data)

    return reduced_data