from pydantic import BaseModel, field_validator
from tutorials.utils.utils import new_section


class MyClass(BaseModel):
    field1: int
    field2: str
    field3: float

    @field_validator("field1")
    def check_field1(cls, value):
        if value < 0 or value > 9:
            raise ValueError("`field1` should be a single digit positive integer.")
        return value


class Settings(BaseModel):
    apli_key: str
    timeout: int = 10


class Item(BaseModel):
    id: int
    name: str
    price: float


if __name__ == "__main__":
    # Instance with matching fields
    new_section("Data class creation")
    my_instance = MyClass(field1=1, field2="Hi!", field3=3.1416)
    print(my_instance)
    print(type(my_instance.field1))
    print(type(my_instance.field2))
    print(type(my_instance.field3))

    # Datatype coercion
    new_section("Data coercion")
    my_new_instance = MyClass(field1=1, field2="Hi!", field3="3")
    print(my_new_instance)
    print(type(my_new_instance.field1))
    print(type(my_new_instance.field2))
    print(type(my_new_instance.field3))

    # Custom Data validation
    new_section("Custom Data validator")
    try:
        my_validated_instance = MyClass(field1=10, field2="Hi!", field3="3")
        print(my_validated_instance)
        print(type(my_validated_instance.field1))
        print(type(my_validated_instance.field2))
        print(type(my_validated_instance.field3))
    except ValueError as e:
        print(f"Validation Error: {str(e)}")

    # Settings Managements
    new_section("Settings manager")
    settings = Settings(apli_key="MYAPIKEY")
    print(settings.apli_key)

    # Serialization
    new_section("Serialization")
    item = Item(id=12, name="Item1", price=29.99)
    print(item)
    json_data = item.model_dump_json()
    print(json_data)

    # Deserialization
    new_section("Deserialization")
    new_json_data = '{"id": 11, "name": "Item2", "price": 59.99}'
    item2 = Item.model_validate_json(new_json_data)
    print(item2)
    print(item2.model_dump())
