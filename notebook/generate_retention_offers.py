import pandas as pd
import random
from datetime import datetime

# -----------------------------------------
# Configuration
# -----------------------------------------

INPUT_FILE = "C:\MY\WORK\PROJECTS\Advanced ds project\Dataset\customers.xlsx"
OUTPUT_FILE = "retention_offers.csv"

df = pd.read_excel(INPUT_FILE)

offers = []

offer_counter = 1

offer_types = [
    "10% Discount",
    "20% Discount",
    "30% Discount",
    "Free Upgrade",
    "Free 3 Months Streaming",
    "Extra 100GB Data",
    "Premium Support",
    "Cashback ₹500",
    "Loyalty Reward",
    "Free Installation"
]

# -----------------------------------------
# Generate Retention Offers
# -----------------------------------------

for _, customer in df.iterrows():

    customer_id = customer["CustomerID"]

    churn = str(customer["Churn Label"]).strip()

    tenure = int(customer["Tenure Months"])

    monthly_charge = float(customer["Monthly Charges"])

    contract = customer["Contract"]

    # -----------------------------------------
    # Offer Probability
    # -----------------------------------------

    if churn == "Yes":

        offer_count = random.randint(1,3)

    elif tenure < 12:

        offer_count = random.randint(0,2)

    else:

        offer_count = random.randint(0,1)

    # -----------------------------------------

    for _ in range(offer_count):

        year = random.randint(2023,2025)
        month = random.randint(1,12)
        day = random.randint(1,28)

        offer_date = datetime(
            year,
            month,
            day
        ).strftime("%Y-%m-%d")

        offer_id = f"OFF{offer_counter:06d}"

        offer_type = random.choice(offer_types)

        # -----------------------------------------
        # Offer Value
        # -----------------------------------------

        if "Discount" in offer_type:

            offer_value = round(
                monthly_charge * random.choice([0.10,0.20,0.30]),
                2
            )

        elif "Cashback" in offer_type:

            offer_value = 500

        elif "100GB" in offer_type:

            offer_value = 100

        else:

            offer_value = 0

        # -----------------------------------------
        # Accepted
        # -----------------------------------------

        if churn == "Yes":

            accepted = random.choices(
                ["Yes","No"],
                weights=[40,60],
                k=1
            )[0]

        else:

            accepted = random.choices(
                ["Yes","No"],
                weights=[70,30],
                k=1
            )[0]

        # -----------------------------------------

        if accepted == "Yes":

            response_date = datetime(
                year,
                month,
                min(day + random.randint(1,7),28)
            ).strftime("%Y-%m-%d")

        else:

            response_date = ""

        # -----------------------------------------
        # Customer Segment
        # -----------------------------------------

        if tenure < 12:

            segment = "New"

        elif tenure < 36:

            segment = "Regular"

        else:

            segment = "Loyal"

        offers.append({

            "offer_id": offer_id,

            "customer_id": customer_id,

            "offer_date": offer_date,

            "offer_type": offer_type,

            "offer_value": offer_value,

            "accepted": accepted,

            "response_date": response_date,

            "customer_segment": segment,

            "churn_label": churn

        })

        offer_counter += 1

# -----------------------------------------
# Save CSV
# -----------------------------------------

offers_df = pd.DataFrame(offers)

offers_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("="*60)
print("retention_offers.csv Generated Successfully")
print("Customers :", len(df))
print("Retention Offers :", len(offers_df))
print("="*60)

print(offers_df.head())