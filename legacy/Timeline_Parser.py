import pandas as pd
from datetime import datetime, timezone
import subprocess
from io import StringIO 
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TemporalDataLoader():
    def __init__(self, dump_path, pid):
        self.dump_path = dump_path
        self.pid = pid

   
    def rawfile_to_df(self):

        ##### sudo python ~/volatility3/vol.py -r csv -f ~/dumps/wmplayer.raw timeliner > ~/Data/timeliner_output.csv ###
        volatility_command = f"sudo python3 /home/yacn/volatility3/vol.py -r csv -f {self.dump_path} timeliner"
        process = subprocess.Popen(volatility_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("Extracting the whole dummp timeline with volatility timeliner plugin")
        stdout, stderr = process.communicate()
        timeliner_output = stdout.decode()
        logging.info("Timeliner Generated" )
        # print(stderr.decode())
        # print("Timeliner output preview:\n", timeliner_output[:500])  # Print the first 500 chars
        try:
            timeliner_df = pd.read_csv(StringIO(timeliner_output))
            # print("Timeliner DataFrame preview:\n", timeliner_df.head())
            return timeliner_df
        except pd.errors.ParserError as e:
            raise RuntimeError("Error parsing CSV:", e)
            
    # Removing the NaN columns and TreeDepth Column 
    def drop_NaN_columns(self, df):
        df.drop(columns='TreeDepth', inplace=True)
        df.dropna(axis='columns', how='all', inplace=True)
        return df

    # Removing the events detected by multiple plugins
    def remove_plugin_duplicates(self, df):
        
        df.drop_duplicates(inplace=True) 
        # Removin same rows with different plugins
        df.drop_duplicates(subset=['Description','Created Date', 'Modified Date'], inplace = True)
        #Removing rows with same "Description" and "Created date" columns but with empty "Modified Date"
        mask = df['Modified Date'].isna() & df.duplicated(subset=['Description', 'Created Date'], keep=False)
        df_cleaned = df[~mask]

        #check the description repetiotions
        repetitions = df_cleaned[df_cleaned.duplicated(subset=['Description'], keep=False)]
        
        return df_cleaned, repetitions

    def sort_dataframe(self, df):
        current_date_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')
        non_thread_events = df[df['Plugin'] != 'ThrdScan']

        # Apply the filtering and event creation process to thread events
        valid_thread_events = df[
            (df['Plugin'] == 'ThrdScan') &
            (df['Modified Date'] > df['Created Date']) &
            (df['Modified Date'] <= current_date_utc)
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


    def filter_pid(self, df):
        return df[df['Description'].str.contains(f"Pid {self.pid} |Process: {self.pid} |Process 6280")]

    def df_to_sequence(self, df):
        sequence = ". ".join(df['Description'].astype(str).to_list())
        return sequence 

    def run(self):
        # df = self.rawfile_to_df()
        df = pd.read_csv("~/Research/Data/CSVs/timeliner_output.csv")
        new_df = self.drop_NaN_columns(df)
        clean_data, repetitions = self.remove_plugin_duplicates(new_df)
        clean_sorted_df = self.sort_dataframe(clean_data)
        filtered_df = self.filter_pid(clean_sorted_df)
        print(len(filtered_df))
        sequence = self.df_to_sequence(filtered_df)
        # repetitions.to_csv("~/Research/Data/CSVs/repetitions.csv")
        # clean_data.to_csv("~/Research/Data/CSVs/clean_data.csv")
        with open("/home/yacn/Research/Data/pid_sequence.txt" , 'w') as file:
            file.write(sequence)

if __name__ == "__main__":

    loader = TemporalDataLoader("/home/yacn/Research/dumps/wmplayer.raw", 6280) 
    loader.run()


