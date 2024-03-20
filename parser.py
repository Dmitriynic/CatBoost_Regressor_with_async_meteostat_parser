import pandas as pd
import aiohttp
import asyncio
from meteostat import Normals, Point

df = pd.read_csv('data/features.csv')
df = df[['lat', 'lon']]

async def get_weather_means(latitude, longitude):
    location = Point(latitude, longitude)
    data = Normals(location, 1961, 1990)
    data = data.fetch()
    means = data.mean()
    print('1')
    return means

async def fetch_data(latitude, longitude):
    async with aiohttp.ClientSession() as session:
        means = await get_weather_means(latitude, longitude)
        return means

async def process_data():
    tasks = []
    for _, row in df.iterrows():
        latitude = row['lat']
        longitude = row['lon']
        task = fetch_data(latitude, longitude)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    return results

loop = asyncio.get_event_loop()
results = loop.run_until_complete(process_data())

for index, result in enumerate(results):
    for column in result.index:
        df.at[index, column] = result[column]

df.to_csv('open_data.csv')