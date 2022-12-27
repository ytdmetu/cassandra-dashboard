# YTD Cassandra Dashboard

For the demo : https://cassandra-dashboard.herokuapp.com/

<img width="1723" alt="Screenshot 2022-12-07 at 21 14 57" src="https://user-images.githubusercontent.com/59481646/206288748-7266581c-ebf1-4c5a-b19c-3a489cddea71.png">
 

  Preconditions:
* Python3
* Pip3
* Cassandra API : https://github.com/ytdmetu/cassandra-api

You need access to Cassandra API to use this service. It should be started in your local machine as pre-request (https://github.com/ytdmetu/cassandra-api/blob/master/README.md)

To use dashboard service, create an Python environment and activate it. Then install project dependencies.
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

In order to run dashboard in local env:
```
python3 app.py
```

The dashboard will be available at `http://127.0.0.1:8050/`

