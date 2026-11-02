import copernicusmarine

copernicusmarine.subset(
    dataset_id="SST_MED_SST_L3S_NRT_OBSERVATIONS_010_012_b",
    username="dnakov",
    password="Dimsaa.24444",
    variables=["adjusted_sea_surface_temperature", "quality_level"],
    minimum_longitude=11.540539899417356,
    maximum_longitude=22.792274020439436,
    minimum_latitude=35.44355236998622,
    maximum_latitude=46.12701709095668,
    start_datetime="2016-06-01T00:00:00",
    end_datetime="2026-06-01T00:00:00",
    output_directory="raw_data/"
)
