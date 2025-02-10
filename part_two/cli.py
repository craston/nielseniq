import typer
import data_handling.splitters as splitters
import main

app = typer.Typer()
app.add_typer(splitters.app, name="split", help="Split data")
app.add_typer(main.app, name="train", help="Main application")

if __name__ == "__main__":
    app()
