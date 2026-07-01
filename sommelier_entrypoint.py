import modal

app = modal.App(name="sommelier")

@app.function()
def square(x: int) -> int:
    print(f"Squaring {x}")
    return x**2

@app.local_entrypoint()
def main():
    print(f"Squaring 42 is: {square.remote(42)}")
