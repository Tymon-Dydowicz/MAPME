import pandas as pd
from src.Config import Settings, PreprocessingSettings

class DataAnalyzer:
    def __init__(self, data: pd.DataFrame, settings: Settings):
        self.data = data
        self.difficultyOrder = settings.difficultyOrder
        self._difficultyOrderLower = [d.lower() for d in self.difficultyOrder]
        self._easiestDifficulty = self._difficultyOrderLower[0] if self.difficultyOrder else None

    def _generateRoomStatistics(self, room: pd.DataFrame) -> dict:
        statistics = {}
        difficulties = room['difficulty'].unique().tolist() if 'difficulty' in room else []
        for difficulty in difficulties:
            if difficulty.lower() not in self._difficultyOrderLower:
                print(f"Warning: Found unexpected difficulty level '{difficulty}' in data. \n",
                      f"Expected levels are: {self.difficultyOrder}.\n"
                        "Unexpected levels may cause incorrect adversarial training.")
        sites = room['site'].unique().tolist() if 'site' in room else []
        
        statistics['number_of_images'] = len(room)
        statistics['difficulties'] = ', '.join(difficulties)
        statistics['sites'] = ', '.join(sites)

        for difficulty in difficulties:
            diff_data = room[room['difficulty'] == difficulty]
            statistics[f'images_{difficulty}'] = len(diff_data)

        for site in sites:
            site_data = room[room['site'] == site]
            statistics[f'images_{site}'] = len(site_data)

        return statistics
    
    def _determineAdversarialTraining(self, rooms: pd.DataFrame) -> bool:
        for index, room in rooms.iterrows():
            difficulties = room["difficulties"].split(", ")
            if self._easiestDifficulty and self._easiestDifficulty not in [d.lower() for d in difficulties]:
                print(f"Room '{room['room']}' is missing the easiest difficulty '{self._easiestDifficulty}'. Skipping adversarial training.")
                return False
            if len(difficulties) < len(self.difficultyOrder):
                print(f"Room '{room['room']}' does not have all difficulty levels. Skipping adversarial training.")
                return False
        return True

    def _determineBalancedClasses(self, rooms: pd.DataFrame) -> bool:
        """
        Determine if room classes are balanced.
        Returns True if balanced, False if some rooms need oversampling.
        """
        minCount = rooms['number_of_images'].min()
        maxCount = rooms['number_of_images'].max()
        
        # TODO include in settings?
        threshold = 0.75
        balanced = minCount >= maxCount * threshold
        suggestedCount = int(maxCount * threshold)
        
        if not balanced:
            for index, room in rooms.iterrows():
                if room['number_of_images'] < suggestedCount:
                    print(f"Room '{room['room']}' is underrepresented: {room['number_of_images']} images. {suggestedCount} suggested for balance.")
            print("Some rooms are underrepresented. Oversampling will be used.")
        
        return balanced


    def analyze(self) -> PreprocessingSettings:
        print("Analyzing data...")
        columns = self.data.columns
        missingValues = self.data.isnull().sum()
        rooms = self.data['room'].unique() if 'room' in columns else []
        roomsStats = []
        for room in rooms:
            roomData = self.data[self.data['room'] == room]
            roomStats = self._generateRoomStatistics(roomData)
            roomStats['room'] = room
            roomsStats.append(roomStats)
            
        roomStats = pd.DataFrame(roomsStats).fillna(0)
        adversarialTraining = self._determineAdversarialTraining(roomStats)
        balancedClasses = self._determineBalancedClasses(roomStats)

        print(f"Data contains the following columns: {columns.tolist()}")
        print(f"Total number of records: {len(self.data)}")
        print(f"Missing values per column:\n{missingValues}")
        print(f"Unique rooms found: {rooms}")
        print(f"Room statistics:\n{roomStats}")

        return PreprocessingSettings(
            adversarialTraining=adversarialTraining,
            oversampling=not balancedClasses
        )

