import typer
import data_handling.splitters as splitters

app = typer.Typer()
app.add_typer(splitters.app, name="split", help="Split data")

if __name__ == "__main__":
    app()
