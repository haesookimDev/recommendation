from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class ComputeSimilarity():
    def __init__(self, df):
        super.__init__
        self.GENDER = df['GENDER']
        self.AGE_GRP = df['AGE_GRP']
        self.M = df['M']
        self.TRAVEL_STATUS_DESTINATION = df['TRAVEL_STATUS_DESTINATION']
        self.TRAVEL_STYL = df['TRAVEL_STYL']
        self.TRAVEL_MOTIVE = df['TRAVEL_MOTIVE']
        self.TRAVEL_PERIOD = df['TRAVEL_PERIOD']
        self.SHOPPING = df['SHOPPING']
        self.PARK = df['PARK']
        self.HISTORY = df['HISTORY']
        self.TOUR = df['TOUR']
        self.SPORTS = df['SPORTS']
        self.ARTS = df['ARTS']
        self.PLAY = df['PLAY']
        self.CAMPING = df['CAMPING']
        self.FESTIVAL = df['FESTIVAL']
        self.SPA = df['SPA']
        self.EDUCATION = df['EDUCATION']
        self.DRAMA = df['DRAMA']
        self.PILGRIMAGE = df['PILGRIMAGE']
        self.WELL = df['WELL']
        self.SNS = df['SNS']
        self.HOTEL = df['HOTEL']
        self.NEWPLACE = df['NEWPLACE']
        self.WITHPET = df['WITHPET']
        self.MIMIC = df['MIMIC']
        self.ECO = df['ECO']
        self.HIKING = df['HIKING']

    def content_based_similarity_traveler(self, all_traveler):
        traveler_vector = np.array([self.TRAVEL_STYL, self.TRAVEL_MOTIVE])
        
        if self.GENDER!=None&self.AGE_GRP!=None:
            filtered_traveler = all_traveler.loc[(all_traveler['GENDER']==self.GENDER)&(all_traveler['AGE_GRP']==self.AGE_GRP)&(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index
        elif self.GENDER!=None&self.AGE_GRP==None:
            filtered_traveler = all_traveler.loc[(all_traveler['GENDER']==self.GENDER)&(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index
        elif self.GENDER==None&self.AGE_GRP!=None:
            filtered_traveler = all_traveler.loc[(all_traveler['AGE_GRP']==self.GENDER)&(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index
        else:
            filtered_traveler = all_traveler.loc[(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
            filtered_traveler_indices = filtered_traveler.index

        all_traveler_vector = np.array([[t['TRAVEL_STYL'], t['TRAVEL_MOTIVE']] for t in filtered_traveler])
        
        
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
        
        filtered_trip = all_trips.loc[((all_trips['M']==self.M))]
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
                                 t['HIKING']] for t in filtered_trip])
        
        trip_similar = cosine_similarity([trip_vector], all_trip_vector)[0]
        
        trip_idx = np.argmax(trip_similar)
        trip_id = filtered_trip_indices[trip_idx]

        return trip_id