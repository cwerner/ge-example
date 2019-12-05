import numpy as np
import pandas as pd
import streamlit as st
import urllib, os

import great_expectations as ge

from minio import Minio
from minio.error import ResponseError

from pathlib import Path

from app_secrets import MINIO_ACCESS_KEY, MINIO_ENCRYPT_KEY

# we use this as data location
BUCKET_NAME = "ge-example"

# pull secrets from a non-tracked secrets file
con = Minio(
    "minio.cwerner.ai",
    access_key=os.getenv("MINIO_ACCESS_KEY", MINIO_ACCESS_KEY),
    secret_key=os.getenv("MINIO_SECRET_KEY", MINIO_ENCRYPT_KEY),
    secure=True,
)

DATE_COLUMN = "date/time"

DATA_YEAR = 2019
DATA_PATH_URL = f"raw/tereno_fendt/{DATA_YEAR}/"
DATA_DEFAULT = "latest.dat"


@st.cache
def load_metadata():
    return open(Path("data") / "colnames.csv").readline().split(",")


@st.cache
def general_ge_check(df):
    # context = ge.data_context.DataContext()
    return None


def parse_jday(files):
    jdays = [int(f.split("_")[-1][:-4]) for f in files if "latest" not in f]
    min_jday = min(jdays)
    min_pos = jdays.index(min_jday)
    max_jday = max(jdays) + 1
    max_pos = -1
    return ((min_jday, max_jday), (min_pos, max_pos))


def load_data(con, bucketname=BUCKET_NAME, date_range=False):

    colnames = load_metadata()

    files = []
    if con.bucket_exists(bucketname):
        print("bucket exists")
        files = con.list_objects(bucketname, prefix=DATA_PATH_URL, recursive=False)
        files = list(files)
    else:
        print("bucket does not exist")
        return None

    ps = [Path(f.object_name).name for f in files]

    n_files = len(files)

    if date_range:
        (min_jday, max_jday), (min_pos, max_pos) = parse_jday(ps)

        sel_dates = st.sidebar.slider(
            "Select date", min_jday, max_jday, (1, 14), step=1
        )

        st.sidebar.success(f"{ps[sel_dates[0]-1][:-4]}-{ps[sel_dates[1]-1][:-4]}")

        dfs = []

        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.sidebar.progress(0)

        sel_files = [files[i] for i in range(sel_dates[0] - 1, sel_dates[1])]
        for i, f in enumerate(sel_files):
            obj = con.get_object(BUCKET_NAME, f.object_name, request_headers=None)

            df = pd.read_csv(
                obj,
                header=None,
                names=colnames,
                index_col="TIMESTAMP",
                parse_dates=["TIMESTAMP"],
            )
            df.index.names = ["index"]
            dfs.append(df)
            bar.progress((i + 1) / len(sel_files))

        df = pd.concat(dfs, axis=0)
        df.sort_index(inplace=True)
        st.sidebar.success(f"Loaded selected files.")

    else:
        sel_file = st.sidebar.selectbox(
            "Select file",
            files,
            format_func=lambda x: Path(x.object_name).name,
            index=n_files - 1,
        )

        obj = con.get_object(BUCKET_NAME, sel_file.object_name, request_headers=None)

        df = pd.read_csv(
            obj,
            header=None,
            names=colnames,
            index_col="TIMESTAMP",
            parse_dates=["TIMESTAMP"],
        )
        df.index.names = ["index"]
        st.sidebar.success(f"Loading file {Path(sel_file.object_name).name}.")

    return df


def main():

    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            "Show instructions",
            "Run the app (1: single)",
            "Run the app (2: multiple)",
            "Show the source code",
        ],
    )
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))
    elif app_mode == "Run the app (1: single)":
        readme_text.empty()
        run_the_app(run_mode="individual")
    elif app_mode == "Run the app (2: multiple)":
        readme_text.empty()
        run_the_app(run_mode="period")


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app(run_mode="individual"):

    date_range = True if run_mode == "period" else False
    df = load_data(con, date_range=date_range)

    st.header("General")

    st.write("... here, we'll show some basic data analysis for all columns...")
    general_ge_check(df)

    st.header("Specific variables")

    var_selection = st.radio("Variables", ["subset", "all"])
    if var_selection == "subset":
        vars = ["airtemp_Avg", "relhumidity_Avg", "airpressure_Avg", "Ramount"]
    else:
        vars = df.columns

    columns = st.multiselect(label="Select column", options=vars)

    st.write("... here, we'll show some specific data analysis")
    st.header("Data summary")

    st.subheader("Raw data")
    st.write(df[columns])

    st.subheader("Plot")
    st.line_chart(df[columns])

    off = """
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_biker": "biker",
            "label_car": "car",
            "label_pedestrian": "pedestrian",
            "label_trafficLight": "traffic light",
            "label_truck": "truck"
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv.gz"))
    summary = create_summary(metadata)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from S3.
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    image = load_image(image_url)

    # Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    draw_image_with_boxes(image, boxes, "Ground Truth",
        "**Human-annotated data** (frame `%i`)" % selected_frame_index)

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    yolo_boxes = yolo_v3(image, overlap_threshold, confidence_threshold)
    draw_image_with_boxes(image, yolo_boxes, "Real-time Computer Vision",
        "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (overlap_threshold, confidence_threshold))



    data_load_state = st.text("Loading data...")
    data = load_data(10000)
    data_load_state.text("Loading data... done!")

    st.sidebar.subheader("Raw data")
    if st.sidebar.checkbox("Show raw data"):
        st.write(data)

    st.subheader("Number of pickups by hour")
    hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
    st.bar_chart(hist_values)

    # Some number in the range 0-23
    hour_to_filter = st.slider("hour", 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

    st.subheader("Map of all pickups at %s:00" % hour_to_filter)
    st.map(filtered_data)"""


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    if Path(path).exists:
        return open(path).read()
    url = "https://raw.githubusercontent.com/cwerner/ge-example/master/" + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


if __name__ == "__main__":
    main()
