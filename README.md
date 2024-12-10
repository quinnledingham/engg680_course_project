**Predicting Wildfires Driven PM2.5 Using Spatiotemporal Transformers in Canada**

_This project aims to predict PM2.5 levels driven by wildfires using spatiotemporal transformers in Canada. The primary goal is to improve air quality forecasting during wildfire events to mitigate public health impacts._

**Project Structure**

├── models/

------│   ├── five_years.model          # Transformer model trained on five years of data

------│   ├── five_years_linear.model   # Linear model trained on five years of data

------│   ├── linear_losses.loss        # Training and validation losses for the Linear model

------│   ├── station_features_v2.csv   # Features used for training the models

------│   ├── transformer_losses.loss   # Training and validation losses for the Transformer model

├── .gitignore                    # Specifies intentionally untracked files to ignore

├── NFDB_point.zip                # Fire data

├── Station Inventory EN.csv      # Initial commit data file

├── StationsNAPS-StationsSNPA.csv # NAPS continuous data

├── StationsNAPS.csv              # NAPS file and distance between coordinates

├── eval.ipynb                    # Evaluation notebook

├── gpt.py                        # Transformer model and related functions

├── input.py                      # Data input and preprocessing functions

├── linear.py                     # Linear model and related functions

├── prepare_data.py               # Prepares data for training

├── naps.py                       # NAPS related functions

├── nfdb.py                       # NFDB related functions

├── train.ipynb                   # Training notebook

**Installation**

To set up the project environment:

**Clone the repository:**

git clone https://github.com/quinnledingham/engg680_course_project.git

**Navigate to the project directory:**

cd engg680_course_project

**Install the dependencies:**

pip install -r requirements.txt

**Usage**

To run the project:

**Data Preprocessing:**

python prepare_data.py

**Model Training:**

Run cells in train.ipynb or python stt.py

**Predicting PM2.5 Levels:**

Run cells in eval.ipynb

_Training: 
The train.ipynb notebook is used for training the Spatiotemporal Transformer model and the Linear model. The models are trained on five years of data, and the trained models and their losses are saved for future evaluation._
Example training outputs:

**Transformer model:**

step 0: train loss 0.24131294, val loss 0.23992151

step 500: train loss 0.00003373, val loss 0.00001246

...

step 4999: train loss 0.00004876, val loss 0.00001857

**Linear model:**

step 0: train loss 0.00017276, val loss 0.00005854

step 500: train loss 0.00001112, val loss 0.00000790

...

step 4999: train loss 0.00000036, val loss 0.00000027

**Evaluation**

The eval.ipynb notebook is used for evaluating the trained models. The notebook loads the pre-trained models, calculates various performance metrics (MSE, R2, MAE, RMSE, MAPE), and visualizes the results.

Key functions in the notebook:

test_error: Evaluates the model on the test set.

gen_24: Generates predictions for the next 24 hours.

train_val_loss_graph: Plots training and validation losses.

plot_scatter: Plots true values vs. predicted values.

plot_comparison_bar_chart: Compares metrics between the Transformer and Linear models.

**Results**

The models were evaluated using multiple metrics, and the results were visualized through various plots.

The Transformer model achieved a lower MSE and higher R2 compared to the Linear model.

Both models showed promising results in predicting PM2.5 levels during wildfire events.

**Contributing**

We welcome contributions to the project. Please submit issues and pull requests to the repository.
