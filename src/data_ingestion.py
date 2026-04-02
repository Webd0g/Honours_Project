import requests as re

url = "https://api.lsst.fink-portal.org/api/v1/tags"
data = {
    'tag': 'sn_near_galaxy_candidate',
    'n': '50'
}

response = re.post(url, json=data)
alerts = response.json()

print("Fink LSST Alert Data:")
print("-" * 30)

for alert in alerts:
    obj_id = alert.get('r:diaObjectId', 'No ID')
    mag = alert.get('r:psfFlux', 'No Flux')
    
    print(f"Object ID: {obj_id}, Flux: {mag}")