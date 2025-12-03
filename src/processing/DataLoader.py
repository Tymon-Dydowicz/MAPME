from src.Config import Settings
from src.Parser import Parser
import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, settings: Settings):
        self.dataPath = settings.dataPath
        self.imageExtensions = settings.imageExtension
        self.dataCsv = settings.dataCsv
        self.namingConvention = settings.imageNamingConvention
        self.parser = Parser(settings)

    def loadData(self) -> pd.DataFrame:
        if not os.path.exists(self.dataPath):
            raise FileNotFoundError(f"Data path {self.dataPath} does not exist.")
        
        if os.path.exists(self.dataCsv):
            data = pd.read_csv(self.dataCsv, dtype=str).fillna('')
            return data

        data = []
        print(f"Loading data from path: {self.dataPath}")
        for root, dirs, files in os.walk(self.dataPath):
            print(f"Current directory: {root}")
            print(f"Subdirectories: {dirs}")
            for file in files:
                print(f"Found file: {file}")
                if file.endswith(tuple(self.imageExtensions)):
                    try:
                        metadata = self.parser.parseFilename(file)
                        metadata = {k: '' if v is None else str(v) for k, v in metadata.items()}
                        metadata['file_path'] = os.path.join(root, file)
                        data.append(metadata)
                    except ValueError as e:
                        print(e)

        df = pd.DataFrame(data)
        df = pd.DataFrame(data).fillna('').astype(str)
        df.to_csv(self.dataCsv, index=False)
        return df


                    
