import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Plant Pathology")


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id:path}")
async def read_user_item(item_id: str, needy: str, skip: int = 0, limit: Optional[int] = None):
    item = {"item_id": item_id, "needy": needy, "skip": skip, "limit": limit}
    return item


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id, "price": item.price}


