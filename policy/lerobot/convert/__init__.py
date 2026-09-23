"""UniVTAC HDF5 episodes -> LeRobot v3 dataset.

* ``hdf5``       reading one episode (joints, camera and fingertip frames)
* ``transforms`` frame/joint transforms shared with the inference side
* ``schema``     dataset keys, feature schema, task instruction, camera choice
* ``pipeline``   the conversion itself
"""
