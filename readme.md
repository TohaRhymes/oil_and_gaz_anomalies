
Main script for anomalies detection:

`anomalies_detect.py`

Launching:

`anomalies_detect.py <data_dir> <images_dir> <data_name>`

Example: `python ./anomalies_detect.py "./data/Фактические-синтетические данные" "./images" "Групповой вылет по воде.xlsx"`



* `six_plots_<X>.jpg` - plots of main characteristics for well №X by days
* `merged_scaled_<X>.jpg` - merged plots of main characteristics of well №X by days, scaled to [0;1]
* `slided_<Y>_six_plots_<X>.jpg` - values are averaged using sliding window with size of Y (+-Y/2 elements) for well №X
* `sub_<Y>_six_plots_<X>.jpg` - windowed values (size=Y) are substracted from absolute values  for well №X.

