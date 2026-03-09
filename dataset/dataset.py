import kagglehub
import shutil

path = kagglehub.dataset_download("sakshigoyal7/credit-card-customers")

destination = r"C:\Users\91914\Desktop\Customer_Spending_&_Engagement_Analytics\dataset"

shutil.move(path, destination)

print("Dataset saved at:", destination)