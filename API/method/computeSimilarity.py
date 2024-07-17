from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class ComputeSimilarity():
    def __init__(self, df: pd.DataFrame):
        super.__init__
        self.GENDER = df['GENDER'].squeeze()
        self.AGE_GRP = df['AGE_GRP'].squeeze()
        self.M = df['M'].squeeze()
        self.TRAVEL_STATUS_DESTINATION = df['TRAVEL_STATUS_DESTINATION'].squeeze()
        self.TRAVEL_STYL = df['TRAVEL_STYL'].squeeze()
        self.TRAVEL_MOTIVE = df['TRAVEL_MOTIVE'].squeeze()
        self.TRAVEL_PERIOD = df['TRAVEL_PERIOD'].squeeze()
        self.SHOPPING = df['SHOPPING'].squeeze()
        self.PARK = df['PARK'].squeeze()
        self.HISTORY = df['HISTORY'].squeeze()
        self.TOUR = df['TOUR'].squeeze()
        self.SPORTS = df['SPORTS'].squeeze()
        self.ARTS = df['ARTS'].squeeze()
        self.PLAY = df['PLAY'].squeeze()
        self.CAMPING = df['CAMPING'].squeeze()
        self.FESTIVAL = df['FESTIVAL'].squeeze()
        self.SPA = df['SPA'].squeeze()
        self.EDUCATION = df['EDUCATION'].squeeze()
        self.DRAMA = df['DRAMA'].squeeze()
        self.PILGRIMAGE = df['PILGRIMAGE'].squeeze()
        self.WELL = df['WELL'].squeeze()
        self.SNS = df['SNS'].squeeze()
        self.HOTEL = df['HOTEL'].squeeze()
        self.NEWPLACE = df['NEWPLACE'].squeeze()
        self.WITHPET = df['WITHPET'].squeeze()
        self.MIMIC = df['MIMIC'].squeeze()
        self.ECO = df['ECO'].squeeze()
        self.HIKING = df['HIKING'].squeeze()

    def content_based_similarity_traveler(self, all_traveler):
        traveler_vector = np.array([self.TRAVEL_STYL, self.TRAVEL_MOTIVE])
        
        if (self.GENDER!=None)&(self.AGE_GRP!=None):
            filtered_traveler: pd.DataFrame = all_traveler.loc[(all_traveler['GENDER']==self.GENDER)&(all_traveler['AGE_GRP']>=self.AGE_GRP-10)&(all_traveler['AGE_GRP']<=self.AGE_GRP+10)&(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index
        elif (self.GENDER!=None)&(self.AGE_GRP==None):
            filtered_traveler: pd.DataFrame = all_traveler.loc[(all_traveler['GENDER']==self.GENDER)&(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index
        elif (self.GENDER==None)&(self.AGE_GRP!=None):
            filtered_traveler: pd.DataFrame = all_traveler.loc[(all_traveler['AGE_GRP']>=self.AGE_GRP-10)&(all_traveler['AGE_GRP']<=self.AGE_GRP+10)&(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index
        else:
            filtered_traveler: pd.DataFrame = all_traveler.loc[(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index


        all_traveler_vector = np.array([[t['TRAVEL_STYL'], t['TRAVEL_MOTIVE']] for idx, t in filtered_traveler.iterrows()])
        
        
        traveler_similar = cosine_similarity([traveler_vector], all_traveler_vector)[0]
        
        traveler_idx = np.argmax(traveler_similar)
        traveler_id = filtered_traveler_indices[traveler_idx]

        return traveler_id
    
    def content_based_similarity_trip(self, all_trips):
        trip_vector = np.array([self.TRAVEL_PERIOD, 
                                self.SHOPPING, 
                                self.PARK, 
                                self.HISTORY, 
                                self.TOUR, 
                                self.SPORTS, 
                                self.ARTS, 
                                self.PLAY, 
                                self.CAMPING, 
                                self.FESTIVAL, 
                                self.SPA, 
                                self.EDUCATION, 
                                self.DRAMA, 
                                self.PILGRIMAGE, 
                                self.WELL, 
                                self.SNS, 
                                self.HOTEL, 
                                self.NEWPLACE, 
                                self.WITHPET, 
                                self.MIMIC, 
                                self.ECO, 
                                self.HIKING])
        
        filtered_trip: pd.DataFrame = all_trips.loc[(all_trips['M']>=self.M-1)&(all_trips['M']<=self.M+1)]
        filtered_trip_indices = filtered_trip.index

        all_trip_vector = np.array([[t['TRAVEL_PERIOD'], 
                                    t['SHOPPING'], 
                                    t['PARK'], 
                                    t['HISTORY'], 
                                    t['TOUR'], 
                                    t['SPORTS'], 
                                    t['ARTS'], 
                                    t['PLAY'], 
                                    t['CAMPING'], 
                                    t['FESTIVAL'],
                                    t['SPA'], 
                                    t['EDUCATION'], 
                                    t['DRAMA'], 
                                    t['PILGRIMAGE'], 
                                    t['WELL'], 
                                    t['SNS'], 
                                    t['HOTEL'], 
                                    t['NEWPLACE'], 
                                    t['WITHPET'], 
                                    t['MIMIC'], 
                                    t['ECO'], 
                                    t['HIKING']] for idx, t in filtered_trip.iterrows()])
        
        trip_similar = cosine_similarity([trip_vector], all_trip_vector)[0]
        
        trip_idx = np.argmax(trip_similar)
        trip_id = filtered_trip_indices[trip_idx]

        return trip_id