from .common import *
import random
import pandas as pd
from scipy.spatial.distance import cdist

US_PLACE_PROMPTS = {
    'empty': '',
    'random': '',
    'coords': 'What are the lat/lon coordinates of ',
    'where_us': 'Where in the United States is ',
}

# zip source https://simplemaps.com/data/us-zips
# county source https://simplemaps.com/data/us-counties
# city source https://simplemaps.com/data/us-cities
# area code source https://github.com/ravisorg/Area-Code-Geolocation-Database


def remove_duplicate_names(place_df, group_name):
    rows_to_keep = []
    for name, group in place_df.sort_values('population', ascending=False).groupby(group_name):
        if len(group) > 1:
            top = group.iloc[0]
            sec = group.iloc[1]
            if top.population > sec.population * 2:
                rows_to_keep.append(group.index[0])
        else:
            rows_to_keep.append(group.index[0])
    return place_df.loc[sorted(rows_to_keep)]


def make_us_place_dataset(test_ratio=0.2):
    # load raw data
    zip_df = pd.read_csv(os.path.join('data', 'raw_data', 'uszips.csv'))
    county_df = pd.read_csv(os.path.join('data', 'raw_data', 'uscounties.csv'))
    city_df = pd.read_csv(os.path.join('data', 'raw_data', 'uscities.csv'))
    college_df = pd.read_csv(os.path.join(
        'data', 'raw_data', 'uscolleges.csv'))

    world_df = pd.read_csv(os.path.join(
        'data', 'entity_datasets', 'world_place.csv'))
    us_landmark_df = world_df.query(
        'country == "United_States" and entity_type != "populated_place"')

    # convert string coords into two columns
    college_lats = []
    college_lons = []
    for str_coord in college_df.coordinates:
        str_coord = str_coord[len('Point('):-1]
        lon, lat = str_coord.split(' ')
        college_lats.append(float(lat))
        college_lons.append(float(lon))

    college_df['latitude'] = college_lats
    college_df['longitude'] = college_lons

    # fill in missing data using nearest neighbors
    get_state_cols = ['state_id', 'state_name', 'timezone', 'lat', 'lng']
    place_df = pd.concat([city_df[get_state_cols], zip_df[get_state_cols]])
    place_coords = place_df[['lng', 'lat']].values
    place_state_ids = place_df['state_id'].values
    place_states = place_df['state_name'].values
    place_timezone = place_df['timezone'].values

    landmark_coords = us_landmark_df[['longitude', 'latitude']].values
    county_coords = county_df[['lng', 'lat']].values
    college_coords = college_df[['longitude', 'latitude']].values
    # get pairwise distances
    place_landmark_dists = cdist(place_coords, landmark_coords)
    place_county_dists = cdist(place_coords, county_coords)
    place_college_dists = cdist(place_coords, college_coords)

    # map landmark to state by min distance to place
    landmark_match = np.argmin(place_landmark_dists, axis=0)
    landmark_states = place_states[landmark_match]
    landmark_state_id = place_state_ids[landmark_match]
    landmark_timezone = place_timezone[landmark_match]

    us_landmark_df['state_id'] = landmark_state_id
    us_landmark_df['state_name'] = landmark_states
    us_landmark_df['timezone'] = landmark_timezone

    # map counties
    county_match = np.argmin(place_county_dists, axis=0)
    county_timezone = place_timezone[county_match]

    county_df['timezone'] = county_timezone

    # map colleges
    college_match = np.argmin(place_college_dists, axis=0)
    college_timezone = place_timezone[college_match]
    college_state_id = place_state_ids[college_match]
    college_state_name = place_states[college_match]

    college_df['timezone'] = college_timezone
    college_df['state_id'] = college_state_id
    college_df['state_name'] = college_state_name

    # Filter out non 48 states
    FILTER_LIST = ['AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP']

    zip_df = zip_df[~zip_df.state_id.isin(FILTER_LIST)]
    county_df = county_df[~county_df.state_id.isin(FILTER_LIST)]
    city_df = city_df[~city_df.state_id.isin(FILTER_LIST)]
    landmark_df = us_landmark_df[~us_landmark_df.state_id.isin(FILTER_LIST)]
    college_df = college_df[~college_df.state_id.isin(FILTER_LIST)]

    # filter duplicates and small places
    county_df = remove_duplicate_names(county_df, 'county_full')
    city_df = remove_duplicate_names(city_df, 'city')

    city_df = city_df.query('population > 500')
    zip_df = zip_df.query(
        '(population > 10000) or (population > 2000 and density > 50)')

    # fix column names
    zip_df['zip'] = zip_df['zip'].astype(str).str.zfill(5)
    zip_df = zip_df.rename(
        columns={'zip': 'name', 'lat': 'latitude', 'lng': 'longitude'})
    city_df = city_df.rename(
        columns={'city': 'name', 'lat': 'latitude', 'lng': 'longitude'})
    county_df = county_df.rename(
        columns={'county_full': 'name', 'lat': 'latitude', 'lng': 'longitude'})
    college_df = college_df.rename(columns={'universityLabel': 'name'})

    city_df['entity_type'] = 'city'
    zip_df['entity_type'] = 'zip'
    county_df['entity_type'] = 'county'
    college_df['entity_type'] = 'college'

    # take only relevant columns
    county_df = county_df[['name', 'latitude', 'longitude', 'state_id',
                           'state_name', 'county_fips', 'population', 'timezone', 'entity_type']]
    city_df = city_df[['name', 'latitude', 'longitude', 'state_id', 'state_name',
                       'county_fips', 'population', 'density', 'timezone', 'entity_type']]
    zip_df = zip_df[['name', 'latitude', 'longitude', 'state_id', 'state_name',
                    'county_fips', 'population', 'density', 'timezone', 'entity_type']]
    landmark_df = landmark_df[['name', 'latitude', 'longitude', 'state_id', 'state_name',
                               'population', 'entity_subtype', 'page_views', 'timezone', 'entity_type']]
    college_df = college_df[['name', 'latitude', 'longitude',
                            'state_id', 'state_name', 'timezone', 'entity_type']]

    us_df = pd.concat([city_df, zip_df, county_df, landmark_df, college_df])
    us_df = us_df.query('latitude < 50')

    # choose test indices (needs to be updated if repeated entities are allowed)
    n = len(us_df)
    test_ixs = np.random.choice(n, size=int(n*test_ratio), replace=False)
    test_set = np.zeros(n, dtype=bool)
    test_set[test_ixs] = True
    us_df['is_test'] = test_set

    us_df.to_csv(os.path.join('data', 'entity_datasets',
                 'us_place.csv'), index=False)


# DEPRECATED
def generate_phone_number(area_code, format):
    prefix = random.randint(100, 999)
    line_number = random.randint(1000, 9999)

    if format == "dash":
        phone_number = "{}-{}-{}".format(area_code, prefix, line_number)
    elif format == "dot":
        phone_number = "{}.{}.{}".format(area_code, prefix, line_number)
    elif format == "space":
        phone_number = "{} {} {}".format(area_code, prefix, line_number)
    elif format == "brackets":
        phone_number = "({}) {}-{}".format(area_code, prefix, line_number)
    elif format == "plain":
        phone_number = "{}{}{}".format(area_code, prefix, line_number)
    else:
        raise ValueError(
            "Invalid format. Choose either 'dash', 'dot', 'space', 'brackets', or 'plain'.")

    return phone_number, prefix, line_number


def make_phone_number_dataset(area_code_df, n_samples=10):
    number_dataset = []
    n_samples = 10
    for ac in area_code_df.area_code.values:
        for format in ["dash", "dot", "space", "brackets", "plain"]:
            for _ in range(n_samples):
                number_str, prefix, line_number = generate_phone_number(
                    ac, format)
                number_dataset.append(
                    [ac, prefix, line_number, format, number_str])

    number_df = pd.DataFrame(
        number_dataset,
        columns=['area_code', 'prefix',
                 'line_number', 'format', 'phone_number']
    )
    number_entity_df = number_df.join(area_code_df.set_index(
        'area_code'), on='area_code', how='left')
    number_entity_df.to_csv(
        'data/entity_datasets/phone_number.csv', index=False)
