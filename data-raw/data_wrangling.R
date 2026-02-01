

data <- readr::read_csv("data-raw/electricity.gui")

data_clean <- electrotidyml::clean_data(data = data)
