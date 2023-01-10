from pydantic import BaseModel, Field

class Page(BaseModel):
    request: str = Field(None, example="")

class PageCreate(UserBase):
    id:int