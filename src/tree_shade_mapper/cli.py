"""Console script for canopy_shade."""
import tree_shade_mapper

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for canopy_shade."""
    console.print("Replace this message by putting your code into "
               "canopy_shade.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
