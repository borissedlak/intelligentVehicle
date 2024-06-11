import requests

# Define Prometheus server URL
prometheus_url = 'http://localhost:9090'

# Define the series selector to delete
query = 'cpu_load'


# Function to delete the series
def delete_series(prometheus_url, query):
    response = requests.post(
        f'{prometheus_url}/api/v1/admin/tsdb/delete_series',
        params={'match[]': query}
    )

    if response.status_code == 200:
        print("Series deleted successfully.")
    else:
        print(f"Error {response.status_code}: {response.text}")


# Function to reload Prometheus configuration
def reload_prometheus_config(prometheus_url):
    response = requests.post(f'{prometheus_url}/-/reload')

    if response.status_code == 200:
        print("Prometheus configuration reloaded successfully.")
    else:
        print(f"Error {response.status_code}: {response.text}")


# Delete the series
delete_series(prometheus_url, query)

# Reload the Prometheus configuration
reload_prometheus_config(prometheus_url)
