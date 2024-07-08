from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class ComputeSimilarity():
    def __init__(self,
                GENDER: int,
                AGE_GRP: int,
                TRAVEL_STATUS_DESTINATION: int,
                M: int,
                TRAVEL_STYL: int,
                TRAVEL_MOTIVE:int,
                TRAVEL_PERIOD: int,
                SHOPPING: int,
                PARK: int,
                HISTORY: int,
                TOUR: int,
                SPORTS: int,
                ARTS: int,
                PLAY: int,
                CAMPING: int,
                FESTIVAL: int,
                SPA: int,
                EDUCATION: int,
                DRAMA: int,
                PILGRIMAGE: int,
                WELL: int,
                SNS: int,
                HOTEL: int,
                NEWPLACE: int,
                WITHPET: int,
                MIMIC: int,
                ECO: int,
                HIKING: int):
        super.__init__
        self.GENDER = GENDER
        self.AGE_GRP = AGE_GRP
        self.M = M
        self.TRAVEL_STATUS_DESTINATION = TRAVEL_STATUS_DESTINATION
        self.TRAVEL_STYL = TRAVEL_STYL
        self.TRAVEL_MOTIVE = TRAVEL_MOTIVE
        self.TRAVEL_PERIOD = TRAVEL_PERIOD
        self.SHOPPING = SHOPPING
        self.PARK = PARK
        self.HISTORY = HISTORY
        self.TOUR = TOUR
        self.SPORTS = SPORTS
        self.ARTS = ARTS
        self.PLAY = PLAY
        self.CAMPING = CAMPING
        self.FESTIVAL = FESTIVAL
        self.SPA = SPA
        self.EDUCATION = EDUCATION
        self.DRAMA = DRAMA
        self.PILGRIMAGE = PILGRIMAGE
        self.WELL = WELL
        self.SNS = SNS
        self.HOTEL = HOTEL
        self.NEWPLACE = NEWPLACE
        self.WITHPET = WITHPET
        self.MIMIC = MIMIC
        self.ECO = ECO
        self.HIKING = HIKING

    def content_based_similarity(self, all_trips, all_traveler):
        traveler_vector = np.array([self.TRAVEL_STYL, self.TRAVEL_MOTIVE])
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
        filtered_traveler = all_traveler.loc[(all_traveler['GENDER']==self.GENDER)&(all_traveler['AGE_GRP']==self.AGE_GRP)(all_traveler['TRAVEL_STATUS_DESTINATION']==self.TRAVEL_STATUS_DESTINATION)]
        filtered_traveler_indices = filtered_traveler.index

        all_traveler_vector = np.array([[t['TRAVEL_STYL'], t['TRAVEL_MOTIVE']] for t in filtered_traveler])
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
        
        traveler_similar = cosine_similarity([traveler_vector], all_traveler_vector)[0]
        trip_similar = cosine_similarity([trip_vector], all_trip_vector)[0]
        
        traveler_idx = np.argmax(traveler_similar)
        traveler_id = filtered_traveler_indices[traveler_idx]
        trip_idx = np.argmax(trip_similar)
        trip_id = filtered_trip_indices[trip_idx]

        return traveler_id, trip_id