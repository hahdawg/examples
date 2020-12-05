import os
import pathlib

src_dir = pathlib.Path(__file__).parent.absolute()
data_dir = os.path.join(src_dir, "data")
df_path = os.path.join(data_dir, "data.hdf")
