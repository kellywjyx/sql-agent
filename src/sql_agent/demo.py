from pathlib import Path
import sqlite3


def create_demo(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
        CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), order_date TEXT, status TEXT);
        CREATE TABLE order_items(order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, unit_price REAL);
        """)
        db.executemany("INSERT INTO customers VALUES(?,?,?)", [(1,"Ada","Singapore"),(2,"Ben","UK"),(3,"Chen","Singapore"),(4,"Dina","Canada")])
        db.executemany("INSERT INTO products VALUES(?,?,?,?)", [(1,"Keyboard","Electronics",80),(2,"Notebook","Stationery",5),(3,"Monitor","Electronics",250),(4,"Pen","Stationery",2)])
        db.executemany("INSERT INTO orders VALUES(?,?,?,?)", [(1,1,"2025-01-03","completed"),(2,2,"2025-01-05","completed"),(3,1,"2025-02-10","cancelled"),(4,3,"2025-02-12","completed"),(5,2,"2025-03-01","completed")])
        db.executemany("INSERT INTO order_items VALUES(?,?,?,?)", [(1,1,1,80),(1,2,3,5),(2,3,1,250),(3,3,1,250),(4,2,10,5),(4,4,5,2),(5,1,2,80)])
