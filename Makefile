plot: test_data/output.csv
	python ../plot.py $<

# Generate output
test_data/output.csv: models/mxs.py models/pyaerso.so test_data
	cd models && python3 mxs.py ../$@

# Create the test_data folder
test_data:
	mkdir -p test_data

# Create the model data
include data_analysis.mk
models/mxs_data.py: $(processed_data) $(analysis_scripts)
	python analysis_scripts/big3.py noplot > $@

# Generate the library
models/pyaerso.so: pyaerso/target/release/libpyaerso.so
	ln -fs ../$< models/pyaerso.so

pyaerso/target/release/libpyaerso.so:
	make -C pyaerso/pytests ../target/release/libpyaerso.so

.PHONY: pyaerso/target/release/libpyaerso.so
