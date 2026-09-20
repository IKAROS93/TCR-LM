# Simulation datasets

These ROS bag files are part of the CARLA-based simulation dataset constructed for open-pit mine shoveling and loading operations.

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1iC8Vz8RGOFY0eBW7DhEvG9BPspIIVBCB?usp=drive_link).

Place the released CARLA bags at the following paths:

```text
datasets/
├── compare_bag1.bag
├── compare_bag2.bag
├── compare_bag3.bag
├── compare_bag4.bag
├── update_bag1.bag
└── gnss_lost_bag1.bag
```

The filename prefixes identify the intended launch scenario. The three launch files resolve these locations relative to `$(find tcrlm)/datasets`.
