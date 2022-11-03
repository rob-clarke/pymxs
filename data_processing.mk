processed_data = \
	wind_tunnel_data/processed_corrected/data_17_10.pkl \
	wind_tunnel_data/processed_corrected/data_17_12.5.pkl \
	wind_tunnel_data/processed_corrected/data_17_15.pkl \
	wind_tunnel_data/processed_corrected/data_17_17.5.pkl \
	wind_tunnel_data/processed_corrected/data_17_20.pkl \
	wind_tunnel_data/processed_corrected/data_17_22.5.pkl \
	wind_tunnel_data/processed_corrected/data_18_10.pkl \
	wind_tunnel_data/processed_corrected/data_18_15.pkl \
	wind_tunnel_data/processed_corrected/data_18_20.pkl

raw_data = wind_tunnel_data/raw/**/*.csv

processing_scripts = \
	processing_scripts/*.py \
	processing_scripts/*.sh \
	processing_scripts/utils/*.py

$(processed_data): $(raw_data) $(processing_scripts)
	processing_scripts/process_november.sh
