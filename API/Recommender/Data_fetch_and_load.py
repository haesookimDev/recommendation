import sys,os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd

from Recommender.Data_processing import DataProcessing

from Recommender.PreprocessingTravelLog import PreprocessingRawData

class DataFetchandLoad():
    def __init__(self, TMA: pd.DataFrame, TA: pd.DataFrame, VAI: pd.DataFrame):
        super().__init__()
        self.TMA=TMA
        self.TA=TA
        self.VAI=VAI
    
    def fetch(self): 
        preprocessingRawData = PreprocessingRawData()

        PRE_TMA, PRE_TA, PRE_VAI = preprocessingRawData.preprocessing(self.TMA, self.TA, self.VAI)

        PRE_TMA.reset_index(drop=True, inplace=True)
        PRE_TA.reset_index(drop=True, inplace=True)
        PRE_VAI.reset_index(drop=True, inplace=True)
        return PRE_TMA, PRE_TA, PRE_VAI
    
    def load(self, TMA: pd.DataFrame, TA: pd.DataFrame, VAI: pd.DataFrame):
        data_processing = DataProcessing()
        data = data_processing.prepare_data(TMA, TA, VAI)

        return data
