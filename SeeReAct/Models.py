from pydantic import BaseModel

class OutputFormat(BaseModel):
    action: str
    element: str
    value: str