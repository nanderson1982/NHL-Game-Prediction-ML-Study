import pandas as pd
import numpy as np
from geopy.distance import geodesic

# NHL cities and Utah with their coordinates (latitude, longitude)
nhl_cities = {
    'Anaheim': (33.807885, -117.876184),
    'Arizona': (33.53193, -112.26101),
    'Boston': (42.360253, -71.058291),
    'Buffalo': (42.880230, -78.878738),
    'Calgary': (51.044734, -114.071883),
    'Carolina': (35.803271, -78.721222),
    'Chicago': (41.878113, -87.629799),
    'Colorado': (39.748667, -105.007512),
    'Columbus': (39.961176, -82.998794),
    'Dallas': (32.776665, -96.796989),
    'Detroit': (42.331427, -83.045754),
    'Edmonton': (53.546124, -113.493823),
    'Florida': (26.158270, -80.325507),
    'Los Angeles': (34.052235, -118.243683),
    'Minnesota': (44.977753, -93.265011),
    'Montreal': (45.501689, -73.567256),
    'Nashville': (36.162664, -86.781602),
    'New Jersey': (40.735657, -74.172367),
    'New York Islanders': (40.722992, -73.590338),
    'New York Rangers': (40.750504, -73.993439),
    'Ottawa': (45.421530, -75.697193),
    'Philadelphia': (39.952583, -75.165222),
    'Pittsburgh': (40.440625, -79.995886),
    'San Jose': (37.338207, -121.886330),
    'Seattle': (47.606209, -122.332069),
    'St. Louis': (38.627003, -90.199404),
    'Tampa Bay': (27.950575, -82.457178),
    'Toronto': (43.651070, -79.347015),
    'Vancouver': (49.282729, -123.120738),
    'Vegas': (36.169941, -115.139830),
    'Washington': (38.907192, -77.036871),
    'Winnipeg': (49.895077, -97.138451),
    'Utah': (40.760780, -111.891045)  # Utah's coordinates (Salt Lake City)
}

# Convert cities to a DataFrame
cities_df = pd.DataFrame(nhl_cities.items(), columns=['City', 'Coordinates'])

# Create a matrix for distances
distance_matrix = pd.DataFrame(index=cities_df['City'], columns=cities_df['City'])

# Calculate the distance between every city pair
for i, city1 in cities_df.iterrows():
    for j, city2 in cities_df.iterrows():
        if city1['City'] != city2['City']:
            distance = geodesic(city1['Coordinates'], city2['Coordinates']).kilometers
            distance_matrix.at[city1['City'], city2['City']] = np.round(distance, 2)

# Display the distance matrix to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="NHL City Distance Matrix with Utah", dataframe=distance_matrix)
