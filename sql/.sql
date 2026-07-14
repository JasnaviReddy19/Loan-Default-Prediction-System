import sqlite3

conn = sqlite3.connect('../sql/loan_data.db')
df.to_sql('loans', conn, if_exists='replace', index=False)

query = """
SELECT Property_Area_Urban, Property_Area_Semiurban,
       AVG(LoanAmount) as avg_loan, COUNT(*) as total,
       SUM(Loan_Status) as approved_count
FROM loans
GROUP BY Property_Area_Urban, Property_Area_Semiurban
"""
result = pd.read_sql(query, conn)
print(result)
conn.close()