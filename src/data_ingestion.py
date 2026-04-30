import requests as re
import pandas as pd
import io

api_url_tags = "https://api.lsst.fink-portal.org/api/v1/tags"
api_url_sources = 'https://api.lsst.fink-portal.org/api/v1/objects'

r0 = re.post(api_url_tags,
    json={
        "tag": "extragalactic_lt20mag_candidate",
        "n": 10000,
        "startdate": "2026-02-19",
        "stopdate": "2026-02-27",
        "columns": "r:diaObjectId",  
    },
)

extra_galactic_candidates = set([str(val["r:diaObjectId"]) for val in r0.json()])
print (f"The list of unique extra galactic candidates is:{extra_galactic_candidates}")
print (f"The number of unique extra galactic candidates is: {len(extra_galactic_candidates)}")

#r1 = re.post(api_url_sources, 
#             json={
#                'columns': 'r:diaObjectId ,r:cats_class, r:u_psfFluxMean, r:psfFluxErr',
#                'output-format': 'json' },
#)

#pdf = pd.read_json(io.BytesIO(r1.content))
#print (pdf)