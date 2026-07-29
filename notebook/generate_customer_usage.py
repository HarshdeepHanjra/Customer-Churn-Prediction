import pandas as pd
import random
from datetime import datetime

INPUT_FILE = "C:\MY\WORK\PROJECTS\Advanced ds project\Dataset\customers.xlsx"
OUTPUT_FILE = "customer_usage.csv"

df = pd.read_excel(INPUT_FILE)

usage_records = []

usage_counter = 1

for _, customer in df.iterrows():

    customer_id = customer["CustomerID"]

    tenure = customer["Tenure Months"]

    if pd.isna(tenure):
        tenure = 1

    tenure = max(1, int(tenure))

    internet_service = customer["Internet Service"]

    phone_service = customer["Phone Service"]

    streaming_tv = customer["Streaming TV"]

    streaming_movies = customer["Streaming Movies"]

    # One usage record per month
    for i in range(tenure):

        year = 2020 + (i // 12)
        month = (i % 12) + 1
        day = random.randint(1, 28)

        usage_date = datetime(
            year,
            month,
            day
        ).strftime("%Y-%m-%d")


        if internet_service == "Fiber optic":

            internet_usage = round(random.uniform(250, 800), 2)

            download_speed = round(random.uniform(120, 600), 2)

            upload_speed = round(random.uniform(60, 250), 2)

        elif internet_service == "DSL":

            internet_usage = round(random.uniform(80, 350), 2)

            download_speed = round(random.uniform(20, 80), 2)

            upload_speed = round(random.uniform(10, 40), 2)

        else:

            internet_usage = 0

            download_speed = 0

            upload_speed = 0


        if phone_service == "Yes":

            call_minutes = random.randint(100, 2500)

            sms_count = random.randint(20, 800)

        else:

            call_minutes = 0

            sms_count = 0


        if streaming_tv == "Yes" or streaming_movies == "Yes":

            streaming_hours = round(
                random.uniform(20, 250),
                2
            )

        else:

            streaming_hours = 0

        usage_records.append({

            "usage_id": f"USE{usage_counter:06d}",

            "customer_id": customer_id,

            "usage_date": usage_date,

            "internet_usage_gb": internet_usage,

            "call_minutes": call_minutes,

            "sms_count": sms_count,

            "streaming_hours": streaming_hours,

            "download_speed": download_speed,

            "upload_speed": upload_speed

        })

        usage_counter += 1


usage_df = pd.DataFrame(usage_records)

usage_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 60)
print("customer_usage.csv Generated Successfully")
print("Customers :", len(df))
print("Usage Records :", len(usage_df))
print("=" * 60)

print(usage_df.head())