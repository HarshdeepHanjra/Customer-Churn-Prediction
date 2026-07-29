import pandas as pd
import random
from datetime import datetime

# ---------------------------------------
# Configuration
# ---------------------------------------

INPUT_FILE = "C:\MY\WORK\PROJECTS\Advanced ds project\Dataset\customers.xlsx"
OUTPUT_FILE = "complaints.csv"

df = pd.read_excel(INPUT_FILE)

complaints = []

complaint_counter = 1

complaint_types = [
    "Poor Network",
    "Slow Internet",
    "Billing Complaint",
    "Wrong Charges",
    "Service Outage",
    "Frequent Disconnections",
    "Poor Customer Support",
    "Installation Delay",
    "Payment Issue",
    "Streaming Issue",
    "Call Drop",
    "Low Download Speed"
]

statuses = [
    "Open",
    "In Progress",
    "Resolved",
    "Closed"
]

departments = [
    "Technical",
    "Billing",
    "Customer Care",
    "Network",
    "Sales"
]

# ---------------------------------------
# Generate Complaints
# ---------------------------------------

for _, customer in df.iterrows():

    customer_id = customer["CustomerID"]

    churn = str(customer["Churn Label"]).strip()

    tech_support = customer["Tech Support"]

    internet = customer["Internet Service"]

    # ---------------------------------------
    # Number of Complaints
    # ---------------------------------------

    if churn == "Yes":

        complaint_count = random.randint(1, 3)

    elif tech_support == "No":

        complaint_count = random.randint(0, 2)

    else:

        complaint_count = random.randint(0, 1)



    for _ in range(complaint_count):

        year = random.randint(2020, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)

        complaint_date = datetime(
            year,
            month,
            day
        ).strftime("%Y-%m-%d")

        complaint_id = f"CMP{complaint_counter:06d}"

        # Complaint Type

        if internet == "No":

            complaint_type = random.choice([
                "Billing Complaint",
                "Payment Issue",
                "Customer Support"
            ])

        else:

            complaint_type = random.choice(complaint_types)


        if churn == "Yes":

            severity = random.choices(
                ["High", "Medium", "Low"],
                weights=[50,35,15],
                k=1
            )[0]

        else:

            severity = random.choices(
                ["High", "Medium", "Low"],
                weights=[15,45,40],
                k=1
            )[0]

        # Status

        status = random.choices(
            statuses,
            weights=[15,20,45,20],
            k=1
        )[0]



        if status in ["Resolved","Closed"]:

            resolution_days = random.randint(1,15)

        else:

            resolution_days = None

        # Customer Satisfaction

        if status in ["Resolved","Closed"]:

            if churn == "Yes":

                satisfaction = random.randint(1,3)

            else:

                satisfaction = random.randint(3,5)

        else:

            satisfaction = None

        complaints.append({

            "complaint_id": complaint_id,

            "customer_id": customer_id,

            "complaint_date": complaint_date,

            "complaint_type": complaint_type,

            "severity": severity,

            "department": random.choice(departments),

            "status": status,

            "resolution_days": resolution_days,

            "customer_satisfaction": satisfaction

        })

        complaint_counter += 1



complaints_df = pd.DataFrame(complaints)

complaints_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("="*60)
print("complaints.csv Generated Successfully")
print("Customers :", len(df))
print("Complaints :", len(complaints_df))
print("="*60)

print(complaints_df.head())