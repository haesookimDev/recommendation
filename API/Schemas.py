from pydantic import BaseModel

class PredictIn(BaseModel):
    traveler_id: int
    GENDER: int
    AGE_GRP: int
    TRAVEL_STATUS_DESTINATION: int
    TRAVEL_STYL: int
    TRAVEL_MOTIVE:int
    trip_id: int
    TRAVEL_PERIOD: int
    SHOPPING: int
    PARK: int
    HISTORY: int
    TOUR: int
    SPORTS: int
    ARTS: int
    PLAY: int
    CAMPING: int
    FESTIVAL: int
    SPA: int
    EDUCATION: int
    DRAMA: int
    PILGRIMAGE: int
    WELL: int
    SNS: int
    HOTEL: int
    NEWPLACE: int
    WITHPET: int
    MIMIC: int
    ECO: int
    HIKING: int

class PredictOut(BaseModel):
    next_destination_id: int
    predicted_rating: float
    predicted_recommend: float
    predicted_revisit: float