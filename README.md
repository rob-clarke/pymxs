# MXS Model

This repository contains the raw data, processing scripts, analysis scripts and model code for a longitudinal flight
dynamics model of an MXS2 airframe. The model is implemented atop `pyaerso`, a Rust-backed Python module for aerodynamic
modelling.

Running `make` in the root will run the processing & analysis scripts, do a demo run of the model and plot the output.

Some of the processing scripts require the `tare` functions from the currently private `BAERigUI` library. Instead of
running these, use the pre-processed `.pkl` files in `wind_tunnel_data/processed_corrected`.
