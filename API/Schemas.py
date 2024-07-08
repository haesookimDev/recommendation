from pydantic import BaseModel

class PredictIn(BaseModel):
    traveler_id: int
    trip_id: int

class PredictOut(BaseModel):
    next_destination_id: int
    predicted_rating: float
    predicted_recommend: float
    predicted_revisit: float