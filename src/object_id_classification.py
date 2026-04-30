import requests as re
import pandas as pd

api_url_sources = 'https://api.lsst.fink-portal.org/api/v1/sources'
object_ids = pd.read_parquet(r"C:\Users\riley\Documents\University\Honours\Honours_Project\data\interim\extragalactic_candidates.parquet")["diaObjectID"].tolist()
test_id = object_ids[0]

def classfication_concatenation( ):
    r0 = re.post(api_url_sources, 
    json = {
        "objectId": test_id,
        "columns": "r:midpointMjdTai,r:psfFlux,r:band",
        "output-format": "json"
        })
    
    r0.raise_for_status()

    classification_df = pd.DataFrame(r0.json())
    print(classification_df.columns.tolist())

classfication_concatenation()