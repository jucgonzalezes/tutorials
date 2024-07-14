from dataclasses import dataclass


@dataclass
class MyClass:
    field1: int
    field2: str
    field3: float


if __name__ == "__main__":
    my_instance = MyClass(1, "Hi!", 3.1416)
    print(my_instance)
    print(type(my_instance.field2))

    my_new_instance = MyClass(1, 2, 3.1416)
    print(my_new_instance)
    print(type(my_new_instance.field2))
