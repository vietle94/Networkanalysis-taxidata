# Networkanalysis-taxidata
Complementary code for network analysis of New York taxi data report

# Instruction:

run network_analysis_project.ipynb 

# Contents:

- network_analysis_project.ipynb: Overview visualization of initial data and the result K-means land usage classification as well as crime prediction

- embedding.py: create the embedding from taxi data. This process takes a long time, result has been saved in data/ZoneEmbed.npy

- test_ml.py: Land usage classification and crime rate prediction

Supporting data:

- crime_counts-2015.npy: total number of crime in each region in one year

- drive_dist.npy: average driving distance between regions

- mh-180.json: 180 regions

- ZoneEmbed.npy: saved result from embedding.py

- taxi_flow_2016-01_2016-02_2016-03.csv light version of data
