# ge-example

This is an experimental data screening toolkit using 
[Great Expectations](https://great-expectations.readthedocs.io/en/latest/)
and [Steamlit](https://streamlit.io). Currently nothing more than a little playground 
app and tech demo for my colleagues here at IMK-IFU.

## Scope

Validate daily data tables from a TERENO field site against some data expectations. 
Expectations are defined using jupyter notebooks and/ or a streamlit web app 
and stored on a minio blob storage instance.

## Technical details

- data blobs are currently stored at minio.cwerner.ai
- streamlit interface currently only used to visualize data

## TODOs

- explore if we can use streamlit to interactively define expectations
  - dropdown for expectation selection
  - plot results of selected expectation on data
- trigger storage of expectations in expectation blob store
