import pandas as pd

class NJCleaner:
    def __init__(self, csv_path : str) -> None:
        self.data = pd.read_csv(csv_path)

    def order_by_scheduled_time(self):
        order = self.data.sort_values(by=['scheduled_time'])
        return order
    
    def prep_df(self, path):
        self.data = self.order_by_scheduled_time()

        #...
        #...
        #...

        self.data.to_csv(path)

    def drop_columns_and_nan(self):
        self.data = self.data.drop(['from', 'to'], axis=1).dropna()
        return self.data
    
    def convert_date_to_day(self):
        self.data['day'] = pd.to_datetime(self.data['date']).dt.day_name()
        self.data.drop(columns=['date'], inplace=True)
        return self.data
    
    def convert_scheduled_time_to_part_of_the_day(self):
        data = self.data.copy()
        
        data['part_of_the_day'] = pd.cut(pd.to_datetime(data['scheduled_time']).dt.hour,
                                         bins=[0, 4, 8, 12, 16, 20, 24],
                                         labels=['late_night', 'early_morning', 'morning',
                                                 'afternoon', 'evening', 'night'],
                                         include_lowest=True)
        data = data.drop('scheduled_time', axis=1)
        return data