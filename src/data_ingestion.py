import requests as re
import pandas as pd
import io

api_url_tags = "https://api.lsst.fink-portal.org/api/v1/tags"
api_url_sources = 'https://api.lsst.fink-portal.org/api/v1/sources'

r0 = re.post(api_url_tags,
    json={
        "tag": "in_tns",
        "n": 10,
        "columns": "r:diaObjectId",  
    },
)

mylist = [str(val["r:diaObjectId"]) for val in r0.json()]

r1 = re.post(api_url_sources, 
             json={
                'diaObjectId': ','.join(mylist),
                'columns': 'r:diaObjectId,r:fink_science_version,r:clf_earlySNIa_score,r:clf_snnSnVsOthers_score',
                'output-format': 'json' },
)

pdf = pd.read_json(io.BytesIO(r1.content))
print (pdf)