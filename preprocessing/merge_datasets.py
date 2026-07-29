import pandas as pd

# ---------------------------------------
# Load Files
# ---------------------------------------

customers = pd.read_excel(r"Dataset/customers.xlsx")

login = pd.read_csv(r"Dataset/login_logs.csv")
transactions = pd.read_csv(r"Dataset/transactions.csv")
usage = pd.read_csv(r"Dataset/customer_usage.csv")
tickets = pd.read_csv(r"Dataset/support_tickets.csv")
complaints = pd.read_csv(r"Dataset/complaints.csv")
emails = pd.read_csv(r"Dataset/email_campaigns.csv")
offers = pd.read_csv(r"Dataset/retention_offers.csv")

# ---------------------------------------
# Login Summary
# ---------------------------------------

login_summary = login.groupby("customer_id").agg(
    total_logins=("login_id", "count"),
    avg_session_minutes=("session_minutes", "mean"),
    successful_logins=("login_status", lambda x: (x == "Success").sum())
).reset_index()

# ---------------------------------------
# Transaction Summary
# ---------------------------------------

transaction_summary = transactions.groupby("customer_id").agg(
    total_transactions=("transaction_id", "count"),
    total_amount=("final_amount", "sum"),
    avg_transaction=("final_amount", "mean")
).reset_index()

# ---------------------------------------
# Usage Summary
# ---------------------------------------

usage_summary = usage.groupby("customer_id").agg(
    total_internet_usage=("internet_usage_gb", "sum"),
    avg_download_speed=("download_speed", "mean"),
    avg_upload_speed=("upload_speed", "mean"),
    total_call_minutes=("call_minutes", "sum"),
    total_sms=("sms_count", "sum"),
    total_streaming_hours=("streaming_hours", "sum")
).reset_index()

# ---------------------------------------
# Support Tickets Summary
# ---------------------------------------

ticket_summary = tickets.groupby("customer_id").agg(
    support_ticket_count=("ticket_id", "count"),
    avg_resolution_time=("resolution_time", "mean"),
    avg_customer_rating=("customer_rating", "mean")
).reset_index()

# ---------------------------------------
# Complaints Summary
# ---------------------------------------

complaint_summary = complaints.groupby("customer_id").agg(
    complaint_count=("complaint_id", "count"),
    avg_resolution_days=("resolution_days", "mean")
).reset_index()

# ---------------------------------------
# Email Summary
# ---------------------------------------

email_summary = emails.groupby("customer_id").agg(
    total_campaigns=("campaign_id", "count"),
    emails_opened=("opened", lambda x: (x == "Yes").sum()),
    emails_clicked=("clicked", lambda x: (x == "Yes").sum()),
    conversions=("conversion", lambda x: (x == "Yes").sum())
).reset_index()

# ---------------------------------------
# Retention Offers Summary
# ---------------------------------------

offer_summary = offers.groupby("customer_id").agg(
    total_offers=("offer_id", "count"),
    offers_accepted=("accepted", lambda x: (x == "Yes").sum()),
    total_offer_value=("offer_value", "sum")
).reset_index()

# ---------------------------------------
# Merge Everything
# ---------------------------------------

merged = customers.copy()

merged = merged.merge(login_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

merged = merged.merge(transaction_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

merged = merged.merge(usage_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

merged = merged.merge(ticket_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

merged = merged.merge(complaint_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

merged = merged.merge(email_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

merged = merged.merge(offer_summary,
                    left_on="CustomerID",
                    right_on="customer_id",
                    how="left").drop(columns=["customer_id"])

# ---------------------------------------
# Replace NaN with 0
# ---------------------------------------

numeric_cols = merged.select_dtypes(include="number").columns
merged[numeric_cols] = merged[numeric_cols].fillna(0)

# ---------------------------------------
# Save
# ---------------------------------------

merged.to_csv("Dataset/merged_dataset.csv", index=False)

print("=" * 60)
print("Merged Dataset Created Successfully")
print("Rows :", merged.shape[0])
print("Columns :", merged.shape[1])
print("=" * 60)

print(merged.head())