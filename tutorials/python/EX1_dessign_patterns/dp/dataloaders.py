from google.cloud import bigquery


def read_from_big_query(query: str):
    client = bigquery.Client()
    query_job = client.query(query)

    results = query_job.result()
    df = results.to_dataframe()

    return df


if __name__ == "__main__":

    query = """
    SELECT pickup_location_id FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2021` 
    GROUP BY 
    pickup_location_id
    ORDER BY pickup_location_id;
    """

    data = read_from_big_query(query)
    print(data)
