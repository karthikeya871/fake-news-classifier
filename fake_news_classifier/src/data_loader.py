import pandas as pd
from config import TRUE_CSV_PATH,FAKE_CSV_PATH
def load_and_merge_data() ->pd.DataFrame:
    """Loads True.csv and Fake.csv and return a combined,shuffled dataFrame"""
    df_true = pd.read_csv(TRUE_CSV_PATH)
    df_false = pd.read_csv(FAKE_CSV_PATH)

    df_true['label']=0
    df_false['label']=1
    df = pd.concat([df_true,df_false] ,ignore_index=True)
    df=df.sample(frac=1.0,random_state=42,).reset_index(drop=True)
    df['full_text']= df['title'].fillna('')+" " + df['text'].fillna('')
    return df