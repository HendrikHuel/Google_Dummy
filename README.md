# Google Dummy Project
This project is a simplified representation of our DS workflow.

# Try the code
To run the code call the main function `main.py`.

# Introduction
	The main function controlls the feature engineering and model estimation pipelines for a single "ek".
    In our real world application we have more then ten "ek"s.

    The pipline is constructed such that: First, all steps depend squential on
    each other in terms of data in and output and, second, all steps can be
    run seperately.

    In total the pipline consists of three steps:
        1. Preprocessing of the .csv files. In our real world application only the .csv's for the most recent date have to be processed.
        2. Pre-calculation of features.
        3. Model estimation

    For our real world application the preprocessing and pre-calculation of
    features runs Thursdays. First the preprocessing updates the newest batch
    of incoming .csv's and there will be a new batch for each "ek". Then the
    feature engineering is triggered. The model estimation is done over the
    weekend since it take up to 39 hours. At the moment all the model estimation
    is done squentialy for each "ek". An the maximal runtime here is 11 hours for a single "ek".

    Each "ek" has several hundreds of these input .csv files in total. On very infrequent
    basis the preprocessing has to be re-run for all files. For example, in te case when a bug is discovered or the preprocessing was
    enhanced. For our biggest "ek"s the processing of all .csv's can take up to four hours using
    parrallelisation where 4 files are processed at once. This is why the
    preprocessing in this dummy app uses dask, because sometimes we need it.


    The flow chart of our process looks as follows:

        file[x].csv's -> <preprocessing> -> file[x].feather's -> <engineering> -> engineered_data.feather -> <model training> -> test_model.pkl

    We use the following data structure.
    The input data for the project is stored at
        0. data/input/ek{str(ek).zfill(3)}
    in .csv format.

    For each of the afromentioned steps, the input data is processed and stored
    during the run of the pipline at different locations as follows:
        1. preprocessed data: data/intermediate/preprocessed/ek{str(ek).zfill(3)}
        2. engineered data: data/intermediate/engineered/f"ek{str(ek).zfill(3)}
        3. models: "data/intermediate/models/f"ek{str(ek).zfill(3)}"

    The models are then used to make predictions at hourly frequency for all "ek"s. This step is omitted in this dummy app to keep things simple.
