import pandas as pd
import random
from datetime import datetime

# -------------------------------------
# Configuration
# -------------------------------------

INPUT_FILE = "C:\MY\WORK\PROJECTS\Advanced ds project\Dataset\customers.xlsx"
OUTPUT_FILE = "email_campaigns.csv"

df = pd.read_excel(INPUT_FILE)

campaigns = []

campaign_counter = 1

campaign_types = [
    "Retention",
    "Promotional",
    "Upgrade",
    "Discount",
    "Loyalty",
    "Win-Back"
]

offer_types = [
    "10% Discount",
    "20% Discount",
    "Free Upgrade",
    "Free Installation",
    "Cashback",
    "Extra Data",
    "Premium Streaming",
    "Loyalty Reward",
    "No Offer"
]

subjects = [
    "Exclusive Offer Just For You",
    "Upgrade Your Plan Today",
    "Limited Time Discount",
    "We Miss You!",
    "Special Loyalty Reward",
    "Save More On Your Next Bill",
    "Free Data Waiting For You",
    "Thank You For Being With Us"
]

# -------------------------------------
# Generate Campaigns
# -------------------------------------

for _, customer in df.iterrows():

    customer_id = customer["CustomerID"]

    churn = str(customer["Churn Label"]).strip()

    tenure = int(customer["Tenure Months"])

    # Number of campaigns
    if churn == "Yes":
        campaign_count = random.randint(2,5)
    else:
        campaign_count = random.randint(3,8)

    for _ in range(campaign_count):

        year = random.randint(2020,2025)
        month = random.randint(1,12)
        day = random.randint(1,28)

        campaign_date = datetime(
            year,
            month,
            day
        ).strftime("%Y-%m-%d")

        campaign_id = f"CMP{campaign_counter:06d}"

        campaign_type = random.choice(campaign_types)

        email_subject = random.choice(subjects)

        email_status = random.choices(
            ["Delivered","Bounced","Spam"],
            weights=[94,3,3],
            k=1
        )[0]

        if email_status == "Delivered":

            if churn == "Yes":

                opened = random.choices(
                    ["Yes","No"],
                    weights=[45,55],
                    k=1
                )[0]

            else:

                opened = random.choices(
                    ["Yes","No"],
                    weights=[75,25],
                    k=1
                )[0]

        else:

            opened = "No"

        if opened == "Yes":

            clicked = random.choices(
                ["Yes","No"],
                weights=[35,65],
                k=1
            )[0]

        else:

            clicked = "No"

        if clicked == "Yes":

            conversion = random.choices(
                ["Yes","No"],
                weights=[25,75],
                k=1
            )[0]

        else:

            conversion = "No"

        offer = random.choice(offer_types)

        campaigns.append({

            "campaign_id": campaign_id,

            "customer_id": customer_id,

            "campaign_date": campaign_date,

            "campaign_type": campaign_type,

            "email_subject": email_subject,

            "email_status": email_status,

            "opened": opened,

            "clicked": clicked,

            "conversion": conversion,

            "offer_type": offer

        })

        campaign_counter += 1

# -------------------------------------
# Save CSV
# -------------------------------------

campaign_df = pd.DataFrame(campaigns)

campaign_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("="*60)
print("email_campaigns.csv Generated Successfully")
print("Customers :", len(df))
print("Campaign Records :", len(campaign_df))
print("="*60)

print(campaign_df.head())