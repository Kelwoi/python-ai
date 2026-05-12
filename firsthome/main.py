import pandas as pd

data = {
    "OrderID": [1001, 1002, 1003],
    "Customer": ["Alice", "Bob", "Alice"],
    "Product": ["Laptop", "Chair", "Mouse"],
    "Category": ["Electronics", "Furniture", "Electronics"],
    "Quantity": [1, 2, 3],
    "Price": [1500, 180, 25],
    "OrderDate": ["2023-06-01", "2023-06-03", "2023-06-05"]
}

df = pd.DataFrame(data)

df["OrderDate"] = pd.to_datetime(df["OrderDate"])

df["TotalAmount"] = df["Quantity"] * df["Price"]

print("Initial table:")
print(df)

print("\nTotal store revenue:")
print(df["TotalAmount"].sum())

print("\nAverage TotalAmount:")
print(df["TotalAmount"].mean())

print("\nNumber of orders for each customer:")
print(df["Customer"].value_counts())

print("\nOrders where TotalAmount is greater than 500:")
print(df[df["TotalAmount"] > 500])

print("\nTable sorted by OrderDate in descending order:")
print(df.sort_values(by="OrderDate", ascending=False))

print("\nOrders from June 5 to June 10 inclusive:")
orders_between_dates = df[
    (df["OrderDate"] >= "2023-06-05") &
    (df["OrderDate"] <= "2023-06-10")
]
print(orders_between_dates)

print("\nGrouped orders by Category:")
category_group = df.groupby("Category").agg(
    ProductsCount=("Quantity", "sum"),
    TotalSales=("TotalAmount", "sum")
)
print(category_group)

print("\nTop 3 customers by total purchase amount:")
top_3_customers = df.groupby("Customer")["TotalAmount"].sum().sort_values(ascending=False).head(3)
print(top_3_customers)