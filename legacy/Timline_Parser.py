import pandas as pd
from datetime import datetime, timezone
import subprocess
from io import StringIO 
import pandas as pd

class TemporalDataLoader():
    def __init__(self, dump_path, pid):
        self.dump_path = dump_path
        self.pid = pid
        self.df = self.rawfile_to_df(self.dump_path)
    
    def rawfile_to_df(self, path):

        ##### sudo python ~/volatility3/vol.py -r csv -f ~/dumps/wmplayer.raw timeliner > ~/Data/timeliner_output.csv ###

        volatility_command = f"sudo python ~/volatility3/vol.py -r csv -f {path} timeliner"
        process = subprocess.Popen(volatility_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if stderr:
            print("Error occurred:", stderr.decode())
        else:
            # Decode the CSV output
            timeliner_output = stdout.decode()
            timeliner_df = pd.read_csv(StringIO(timeliner_output))
            print(timeliner_df.head())

        return timeliner_df


    # Removing the NaN columns and TreeDepth Column 
    def drop_NaN_columns(self):
        self.df.drop(columns='TreeDepth', inplace=True)
        self.df.dropna(axis='columns', how='all', inplace=True)
        # self.df.drop(self.df['Modified Date'] > )
        # self.df = self.df[self.df['Created Date'] < self.df['Modified Date']]
        return self.df

    # Removing the events detected by multiple plugins
    def remove_plugin_duplicates(self):
        
        self.df.drop_duplicates(inplace=True) 
        # Removin same rows with different plugins
        self.df.drop_duplicates(subset=['Description','Created Date', 'Modified Date'], inplace = True)
        #Removing rows with same "Description" and "Created date" columns but with empty "Modified Date"
        mask = self.df['Modified Date'].isna() & self.df.duplicated(subset=['Description', 'Created Date'], keep=False)
        df_cleaned = self.df[~mask]

        #check the description repetiotions
        repetitions = df_cleaned[df_cleaned.duplicated(subset=['Description'], keep=False)]
        
        return df_cleaned, repetitions

    def sort_dataframe(self):
        current_date_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')
        non_thread_events = self.df[self.df['Plugin'] != 'ThrdScan']

        # Apply the filtering and event creation process to thread events
        valid_thread_events = self.df[
            (self.df['Plugin'] == 'ThrdScan') &
            (self.df['Modified Date'] > self.df['Created Date']) &
            (self.df['Modified Date'] <= current_date_utc)
        ]

        thread_event_list = []
        # Iterate through the filtered rows to generate 'created' and 'modified' events without the plus sign
        for index, row in valid_thread_events.iterrows():
            # Add 'created' event
            thread_event_list.append({
                'Description': row['Description'] + ' created', 
                'Date': row['Created Date']
            })
            
            # Add 'modified' event
            thread_event_list.append({
                'Description': row['Description'] + ' modified',
                'Date': row['Modified Date']
            })

        # Combine the thread events with the non-thread events
        threads_events_df = pd.DataFrame(thread_event_list)
        combined_events_df = pd.concat([threads_events_df, non_thread_events[['Description', 'Created Date']].rename(columns={'Created Date': 'Date'})])
        # Sort all events by date
        combined_events_df = combined_events_df.sort_values(by='Date').reset_index(drop=True)

        return combined_events_df


    def filter_pid(self):
        return self.df[self.df['Description'].str.contains(f"pid {self.pid} |process {self.pid} ")]

    def df_to_sequence(self):
        sequence = ". ".join(self.df['Description'].astype(str).to_list())
        return sequence 

    def run(self):
        # df = pd.read_csv("~/Research/Data/CSVs/timeliner_output.csv")
        new_df = self.drop_NaN_columns(self.df)
        clean_data, repetitions = self.remove_plugin_duplicates(new_df)
        clean_sorted_df = self.sort_dataframe(clean_data)
        sequence = self.df_to_sequence(clean_sorted_df)
        # repetitions.to_csv("~/Research/Data/CSVs/repetitions.csv")
        # clean_data.to_csv("~/Research/Data/CSVs/clean_data.csv")
        # clean_sorted_df.to_csv("~/Research/Data/CSVs/sorted_clean_df.csv")
        with open("/home/yacn/Research/Data/final_sequence.txt" , 'w') as file:
            file.write(sequence)

if __name__ == "__main__":

    loader = TemporalDataLoader() 
    TemporalDataLoader.run()
    

    # # rawfile_to_df("~/Research/wmplayer.raw")
    #     df = pd.read_csv("~/Research/Data/CSVs/timeliner_output.csv")
    #     new_df = drop_NaN_columns(df)
    #     clean_data, repetitions = remove_plugin_duplicates(new_df)
    #     clean_sorted_df = sort_dataframe(clean_data)
    #     sequence = df_to_sequence(clean_sorted_df)
    #     repetitions.to_csv("~/Research/Data/CSVs/repetitions.csv")
    #     clean_data.to_csv("~/Research/Data/CSVs/clean_data.csv")
    #     clean_sorted_df.to_csv("~/Research/Data/CSVs/sorted_clean_df.csv")
    #     with open("/home/yacn/Research/Data/final_sequence.txt" , 'w') as file:
    #         file.write(sequence)



#################################################################################

# Sample DataFrame
    # data = {
    #     'plugin': ["thread", "thread", "DLL", "DLL"],
    #     'Description': ["pid 18 in offset 654818", "pid 4 in offset 4567098", "process 180 in offset 2345678", "process 18 in offset 9876879"],
    #     'time': [9, 10, 11, 12]
    # }
    # df = pd.DataFrame(data)
    # print(df)
    # pid = 18
    # print(df[df['Description'].str.contains(f"pid {pid} |process {pid} ")])
    # for index,row in df.iterrows():
    #     print(4 in row.values)
    # new_df = filter_pid(df,18)
    # print(new_df)