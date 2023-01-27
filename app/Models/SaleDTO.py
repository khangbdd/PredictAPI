from pydantic import BaseModel

class SaleDTO(BaseModel):
    date: str
    sale: str