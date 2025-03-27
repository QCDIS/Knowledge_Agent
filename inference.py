from transformers import AutoModelForCausalLM, AutoTokenizer


# Load fine-tuned model

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
trained_model_name = "./fine_tuned_model"

#model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    trained_model_name,
    load_in_4bit=True,  # Load model in 4-bit precision for lower memory usage
    device_map="auto"  # Automatically assign layers to GPUs
)

tokenizer = AutoTokenizer.from_pretrained(trained_model_name)
print("Model and tokenizer loaded ...")
# Generate text using the fine-tuned model
input_text = "Wind and wave data from North Sea Platforms"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
print("Generating response ...")
output = model.generate(**inputs, max_length=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))


# Trained Model: 
# Wind and wave data from North Sea Platforms
"""
####### Causal Training Instruction Model ###########

Wind and wave data from North Sea Platforms
The dataset contains wind and wave data from the North Sea, measured on platforms in the North Sea.
The dataset is a compilation of data from several platforms, including the Ekofisk, Frigg, and Statfjord platforms. The data were collected over a period of several years, and the dataset contains a large number of records, each representing a single measurement.
The data are organized into several categories, including:
Wind speed and direction
Significant wave height
Peak wave period
The dataset also includes some additional information, such as the date and time of each measurement, and the location of the platform where the measurement was taken.
The data are in CSV format, with each record representing a single measurement. The dataset is available for download from the Norwegian Petroleum Directorate's website.
The dataset is useful for a variety of applications, including:
Research into the wind and wave climate of the North Sea
Planning and design of offshore oil and gas platforms

####### SEQ2SEQ Training Instruction Model ###########
Wind and wave data from North Sea Platforms
This dataset contains wind and wave data from North Sea platforms. The data are collected from various platforms located in the North Sea, including the Ekofisk, Frigg, and Statfjord platforms. The data are collected from 1995 to 2006. The data are in CSV format.
The data are collected from anemometers and wave meters located on the platforms. The anemometers measure wind speed and direction, while the wave meters measure wave height and period. The data are collected at 10-minute intervals.
The dataset contains 12 variables, including:
- Wind speed (m/s)
- Wind direction (degrees)
- Wave height (m)
- Wave period (s)
- Date
- Time
- Platform name
- Platform location
- Platform depth
- Platform water depth
- Platform latitude
- Platform longitude
The data are collected from various platforms, including:
- Ekofisk
- Frigg


##########  SEQ2SEQ on meta-llama/Llama-2-7b-hf ##########

Wind and wave data from North Sea Platforms (1974-1987)
ϊ Wind and wave data from Brent Platform (1974-1987)
Data holding period: Various periods between 1975 and 1987
Data holding organisation: Shell Exploration and Production
Data holding country: United Kingdom
Data holding centre: UK Offshore Operators Association
The data set comprises various measurements of winds and waves, mostly collected by Marex (now Paras), on behalf of Shell. Wind data from Brent Platform and wind and wave data from North Cormorant were gathered by Shell.
Wind and wave data from North Sea Platforms and wind data from Brent Platform were gathered by Paras on behalf of various operators.
Data from North Cormorant and data from Brent Platform and North Cormorant Parasowed


##########  SEQ2SEQ on meta-llama/Meta-Llama-3.1-8B-Instruct ##########

Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind and wave data from North Sea Platforms: 1974-1987
Cited by 112
Wind

"""

# Base Model: 
"""
Wind and wave data from North Sea Platforms
The data is a collection of wind and wave measurements from the North Sea, specifically from the platforms located in the North Sea. The data is recorded at 10-minute intervals and includes measurements of wind speed, wind direction, and wave height. The data is provided in CSV format.
The data is a collection of wind and wave measurements from the North Sea, specifically from the platforms located in the North Sea. The data is recorded at 10-minute intervals and includes measurements of wind speed, wind direction, and wave height. The data is provided in CSV format.
The data is a collection of wind and wave measurements from the North Sea, specifically from the platforms located in the North Sea. The data is recorded at 10-minute intervals and includes measurements of wind speed, wind direction, and wave height. The data is provided in CSV format.
The data is a collection of wind and wave measurements from the North Sea, specifically from the platforms located in the
"""