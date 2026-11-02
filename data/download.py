import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_obs-oc_med_bgc-plankton_my_l3-multi-1km_P1D",
  username="dnakov",
  password="Dimsaa.24444",
  variables=["CHL"],
  minimum_longitude=11.540539899417356,
  maximum_longitude=22.792274020439436,
  minimum_latitude=35.44355236998622,
  maximum_latitude=46.12701709095668,
  start_datetime="2016-06-01T00:00:00",
  end_datetime="2026-06-01T00:00:00",
  output_directory="raw_data/"
)
