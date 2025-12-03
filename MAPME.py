import pandas as pd
from src.Config import Settings, settings
from src.DataLoader import DataLoader
from src.DataAnalyzer import DataAnalyzer
from src.DataProcessor import DataProcessor


def loadSettings():
    return settings

def loadData(settings: Settings) -> pd.DataFrame:
    dataLoader = DataLoader(settings)
    data = dataLoader.loadData()
    return data
    
def analyzeData(data: pd.DataFrame, settings: Settings):
    analyzer = DataAnalyzer(data, settings)
    return analyzer.analyze()

def main():
    settings = loadSettings()
    print("Current Configuration:")
    print(settings.describe())
    data = loadData(settings)
    preprocessingSettings = analyzeData(data, settings)
    print("Preprocessing Settings:")
    print(preprocessingSettings.describe())

    dataProcessor = DataProcessor(data, preprocessingSettings)
    processed_data = dataProcessor.process()
    print(processed_data.head())

if __name__ == "__main__":
    main()